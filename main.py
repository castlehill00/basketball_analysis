import os
import json
import argparse
from copy import deepcopy
from utils import VideoFrames, save_video
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from camera_cut_detector import CameraCutDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter, CourtLineRefiner
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from shot_detector import ShotDetector
from drawers import (
    PlayerTracksDrawer, 
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer,
    CourtLinesDrawer,
    ShotDrawer
)
from configs import(
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)

# The ball is drawn, and used for possession, where it was detected and between two detections at
# most this many frames apart in the same camera shot (filling longer gaps puts it in the wrong place)
BALL_MAX_INTERPOLATION_GAP = 6

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=None,
                        help='Path to stub directory (default: a folder named after the video under '
                             f'{STUBS_DEFAULT_PATH}/, so that different videos never share stubs)')
    parser.add_argument('--league', type=str, default='fiba', choices=['fiba', 'nba'],
                        help='Court dimensions: fiba (also the CBA) or nba')
    return parser.parse_args()

def main():
    args = parse_args()
    stub_path = args.stub_path or os.path.join(STUBS_DEFAULT_PATH, os.path.splitext(os.path.basename(args.input_video))[0])

    # Read Video (frames are decoded when they are used, so long videos fit in memory)
    video_frames = VideoFrames(args.input_video)
    
    ## Initialize Tracker
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)

    ## Initialize Keypoint Detector
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    # Run Detectors
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=os.path.join(stub_path, 'player_track_stubs.pkl')
                                      )
    
    ## The ball model also detects the hoop, so both come from one pass
    ball_tracks, hoop_tracks = ball_tracker.get_tracks_of_classes(video_frames,
                                                                  ['Ball', 'Hoop'],
                                                                  read_from_stub=True,
                                                                  stub_paths=[os.path.join(stub_path, 'ball_track_stubs.pkl'),
                                                                              os.path.join(stub_path, 'hoop_track_stubs.pkl')]
                                                                 )
    ## Run KeyPoint Extractor
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                    read_from_stub=True,
                                                                    stub_path=os.path.join(stub_path, 'court_key_points_stub.pkl')
                                                                    )
    ## Find the camera shots: the court homography is checked and smoothed within each one
    camera_shots = CameraCutDetector().get_segments(video_frames,
                                                    read_from_stub=True,
                                                    stub_path=os.path.join(stub_path, 'camera_cut_stub.pkl')
                                                    )

    # Remove Wrong Ball Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks, camera_shots=camera_shots)
    # Shot detection needs the ball without long interpolated gaps, so keep a copy
    filtered_ball_tracks = deepcopy(ball_tracks)
    # Interpolate Ball Tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks, camera_shots, max_gap=BALL_MAX_INTERPOLATION_GAP)
   

    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                    player_tracks,
                                                                    read_from_stub=True,
                                                                    stub_path=os.path.join(stub_path, 'player_assignment_stub.pkl')
                                                                    )

    # Ball Acquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks,camera_shots)

    # Detect Passes
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)

    # Tactical View
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png",
        league=args.league
    )

    # validate_keypoints is no longer used: its distance-ratio check ignores perspective and drops
    # good keypoints; RANSAC in get_homographies takes care of wrong ones
    court_keypoints_per_frame = tactical_view_converter.filter_low_confidence_keypoints(court_keypoints_per_frame)
    ## Fit the court to the keypoints and check and smooth it within each camera shot, then adjust
    ## it to the painted court lines (cached, as this decodes the video again)
    court_homographies = CourtLineRefiner(tactical_view_converter).get_homographies(
        video_frames, court_keypoints_per_frame, camera_shots,
        read_from_stub=True, stub_path=os.path.join(stub_path, 'court_line_refinement_stub.pkl'))
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_homographies,player_tracks)

    # Detect Shots and classify them as two- or three-pointers
    shot_detector = ShotDetector(tactical_view_converter)
    shots = shot_detector.detect_shots(filtered_ball_tracks, hoop_tracks, player_tracks, court_homographies,
                                       player_assignment)
    shots_path = os.path.splitext(args.output_video)[0] + '_shots.json'
    os.makedirs(os.path.dirname(shots_path) or '.', exist_ok=True)
    with open(shots_path, 'w') as f:
        json.dump(shots, f, indent=1)

    # Speed and Distance Calculator
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)

    # Draw output (the drawers are generators: each frame goes through all of them and is
    # written before the next one is read)
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()
    court_lines_drawer = CourtLinesDrawer()
    shot_drawer = ShotDrawer()

    ## Draw object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, 
                                                    player_tracks,
                                                    player_assignment,
                                                    ball_aquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    ## Draw KeyPoints (only those that agree with the court homography)
    output_video_frames = court_keypoint_drawer.draw(output_video_frames,
                                                     court_keypoints_per_frame,
                                                     court_homographies,
                                                     tactical_view_converter)

    ## Draw Court Lines
    output_video_frames = court_lines_drawer.draw(output_video_frames,
                                                  court_homographies,
                                                  tactical_view_converter.court_polylines)

    ## Draw Shots
    output_video_frames = shot_drawer.draw(output_video_frames,
                                           shots,
                                           court_homographies,
                                           tactical_view_converter)

    ## Draw Frame Number
    output_video_frames = frame_number_drawer.draw(output_video_frames)

    # Draw Team Ball Control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                        player_assignment,
                                                        ball_aquisition)

    # Draw Passes and Interceptions
    output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
                                                             passes,
                                                             interceptions)
    
    # Speed and Distance Drawer
    output_video_frames = speed_and_distance_drawer.draw(output_video_frames,
                                                         player_tracks,
                                                         player_distances_per_frame,
                                                         player_speed_per_frame
                                                         )

    ## Draw Tactical View
    output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                    tactical_view_converter.court_image_path,
                                                    tactical_view_converter.width,
                                                    tactical_view_converter.height,
                                                    tactical_view_converter.key_points,
                                                    tactical_player_positions,
                                                    player_assignment,
                                                    ball_aquisition,
                                                    )

    # Save video
    save_video(output_video_frames, args.output_video, fps=video_frames.fps)

if __name__ == '__main__':
    main()
    