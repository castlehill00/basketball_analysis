"""
Compares homography settings of TacticalViewConverter on the sample videos.

Run from the repository root after main.py has produced the stubs of each video:

    python evaluation/eval_homography.py --videos 1 2 3 --figure_dir ../report_figures

There is no ground truth for where players stand, so the settings are compared on signals
that a correct homography has to satisfy:
    coverage      share of frames that get a homography
    in court      share of player foot points that land on the court (1 m margin)
    speed         player speed from court positions 10 frames apart; above 36 km/h is faster
                  than any basketball player runs
    agreement     share of confident detected keypoints that the homography puts within 10 px
                  of where they were detected (labels may be mirrored to the other half, since
                  the detector mixes up the halves and that is measured by the speed instead)
    jitter        second difference over consecutive frames of where fixed image points land on
                  the court; a smoothly moving camera gives close to 0

The last two settings add the stricter checks of each frame's fit (at least five keypoints
agreeing, within 0.2 m) and the checking and smoothing per camera shot; evaluation/eval_court.py
measures those on a labelled game.
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tactical_view_converter import TacticalViewConverter
from camera_cut_detector import CameraCutDetector
from drawers import CourtLinesDrawer, TacticalViewDrawer
from utils import get_foot_position, read_video, VideoFrames

# name, keypoint confidence threshold, run validate_keypoints, RANSAC threshold (m),
# fix mirrored labels, smoothing window, then optionally the stricter frame checks and the
# smoothing per camera shot (both off when left out)
SETTINGS = [
    ('all keypoints', None, True, None, False, 1),
    ('original (conf>=0.5)', 0.5, True, None, False, 1),
    ('+RANSAC', 0.5, True, 1.0, False, 1),
    ('+RANSAC -validate', 0.5, False, 1.0, False, 1),
    ('+mirror fix', 0.5, False, 1.0, True, 1),
    ('+smoothing 3', 0.5, False, 1.0, True, 3),
    ('+smoothing 9', 0.5, False, 1.0, True, 9),
    ('+smoothing 5', 0.5, False, 1.0, True, 5),
    ('+stricter frame checks', 0.5, False, 1.0, True, 5, True),
    ('+camera shots (final)', 0.5, False, 1.0, True, 5, True, True),
]
MAX_PLAYER_SPEED_KMH = 36
CONFIDENT_KEYPOINT = 0.5
AGREEMENT_PIXELS = 10


def to_numpy(values):
    return values.cpu().numpy() if hasattr(values, 'cpu') else np.asarray(values)


def get_homographies(keypoints, league, camera_shots, conf_threshold, validate, ransac_threshold, fix_mirrored_labels,
                     smoothing_window, strict_frame_checks=False, use_camera_shots=False):
    frame_checks = {} if strict_frame_checks else {'min_inliers': 4, 'max_inlier_error_in_meters': None}
    converter = TacticalViewConverter('./images/basketball_court.png',
                                      league=league,
                                      keypoint_conf_threshold=conf_threshold,
                                      ransac_threshold_in_meters=ransac_threshold,
                                      fix_mirrored_labels=fix_mirrored_labels,
                                      smoothing_window=smoothing_window,
                                      **frame_checks)
    keypoints = converter.filter_low_confidence_keypoints(keypoints)
    if validate:
        keypoints = converter.validate_keypoints(keypoints)
    return converter, converter.get_homographies(keypoints, camera_shots if use_camera_shots else None)


def collect_measurements(converter, homographies, raw_keypoints, player_tracks, fps, lag):
    """Returns the raw measurements of one video so they can be pooled across videos."""
    meters_per_pixel = np.array([converter.actual_width_in_meters / converter.width,
                                 converter.actual_height_in_meters / converter.height])
    margin = converter.meters_to_pixels(1.0)

    positions = []
    in_court = []
    for homography, frame_tracks in zip(homographies, player_tracks):
        frame_positions = {}
        if homography is not None and frame_tracks:
            player_ids = list(frame_tracks)
            feet = np.array([[get_foot_position(frame_tracks[i]['bbox'])] for i in player_ids], dtype=np.float32)
            mapped = cv2.perspectiveTransform(feet, homography).reshape(-1, 2)
            for player_id, (x, y) in zip(player_ids, mapped):
                inside = -margin <= x <= converter.width + margin and -margin <= y <= converter.height + margin
                in_court.append(inside)
                if inside:
                    frame_positions[player_id] = np.array([x, y]) * meters_per_pixel
        positions.append(frame_positions)

    speeds = []
    for frame_idx in range(len(positions) - lag):
        for player_id, position in positions[frame_idx].items():
            later_position = positions[frame_idx + lag].get(player_id)
            if later_position is not None:
                speeds.append(np.linalg.norm(later_position - position) / (lag / fps) * 3.6)

    template = np.array(converter.key_points, dtype=np.float32).reshape(-1, 1, 2)
    mirrored = converter.mirrored_key_point_indices
    keypoint_agreement = []
    for homography, frame_keypoints in zip(homographies, raw_keypoints):
        if homography is None or len(frame_keypoints.xy) == 0:
            continue
        xy = to_numpy(frame_keypoints.xy)[0]
        confident = np.flatnonzero((to_numpy(frame_keypoints.conf)[0] >= CONFIDENT_KEYPOINT) & (xy[:, 0] > 0) & (xy[:, 1] > 0))
        if confident.size == 0:
            continue
        projected = cv2.perspectiveTransform(template, np.linalg.inv(homography)).reshape(-1, 2)
        as_labelled = np.linalg.norm(projected[confident] - xy[confident], axis=1) <= AGREEMENT_PIXELS
        as_mirrored = np.linalg.norm(projected[[mirrored[i] for i in confident]] - xy[confident], axis=1) <= AGREEMENT_PIXELS
        keypoint_agreement.extend(as_labelled if as_labelled.sum() >= as_mirrored.sum() else as_mirrored)

    frame_height, frame_width = raw_keypoints[0].orig_shape
    grid = np.array([[[x * frame_width, y * frame_height]] for x in (0.2, 0.5, 0.8) for y in (0.6, 0.8, 0.95)],
                    dtype=np.float32)
    projected_grids = [None if h is None else cv2.perspectiveTransform(grid, h).reshape(-1, 2) for h in homographies]
    jitter = []
    for frame_idx in range(1, len(projected_grids) - 1):
        previous, current, following = projected_grids[frame_idx - 1:frame_idx + 2]
        if previous is None or current is None or following is None:
            continue
        on_court = ((current[:, 0] >= -margin) & (current[:, 0] <= converter.width + margin) &
                    (current[:, 1] >= -margin) & (current[:, 1] <= converter.height + margin))
        second_difference = (following - 2 * current + previous)[on_court] * meters_per_pixel
        jitter.extend(np.linalg.norm(second_difference, axis=1))

    return {
        'frames': len(homographies),
        'frames_with_homography': sum(h is not None for h in homographies),
        'in_court': in_court,
        'speeds': speeds,
        'keypoint_agreement': keypoint_agreement,
        'jitter': jitter,
    }


def summarize(measurements):
    speeds = np.array(measurements['speeds'])
    jitter = np.array(measurements['jitter'])
    return {
        'coverage': measurements['frames_with_homography'] / measurements['frames'],
        'in_court': np.mean(measurements['in_court']) if measurements['in_court'] else float('nan'),
        'speed_p95': np.percentile(speeds, 95) if speeds.size else float('nan'),
        'implausible_speed': np.mean(speeds > MAX_PLAYER_SPEED_KMH) if speeds.size else float('nan'),
        'keypoint_agreement': np.mean(measurements['keypoint_agreement']) if measurements['keypoint_agreement'] else float('nan'),
        'jitter_median': np.median(jitter) if jitter.size else float('nan'),
        'jitter_p95': np.percentile(jitter, 95) if jitter.size else float('nan'),
    }


def merge(measurements_list):
    merged = {'frames': 0, 'frames_with_homography': 0, 'in_court': [], 'speeds': [], 'keypoint_agreement': [], 'jitter': []}
    for measurements in measurements_list:
        for key in merged:
            merged[key] += measurements[key]
    return merged


def print_table(title, rows):
    print(f'\n## {title}')
    print('| setting | coverage | in court | speed > 36 km/h | speed p95 (km/h) | keypoint agreement | jitter median / p95 (m) |')
    print('|---|---|---|---|---|---|---|')
    for name, s in rows:
        print(f"| {name} | {s['coverage']:.0%} | {s['in_court']:.1%} | {s['implausible_speed']:.1%} | "
              f"{s['speed_p95']:.1f} | {s['keypoint_agreement']:.1%} | {s['jitter_median']:.2f} / {s['jitter_p95']:.2f} |")


def save_comparison_figures(video_idx, frames, player_tracks, player_assignment, converter, before, after,
                            figure_dir, count=2):
    """
    Saves side-by-side court line overlays and tactical views for the frames where the two
    settings disagree most. The tactical view matters: a homography for the wrong half of the
    court draws plausible lines on the frame (the court is symmetric) but puts every player on
    the other half.
    """
    frame_height, frame_width = frames[0].shape[:2]
    grid = np.array([[[x * frame_width, y * frame_height]] for x in (0.2, 0.5, 0.8) for y in (0.5, 0.75, 1.0)],
                    dtype=np.float32)
    meters_per_pixel = converter.actual_width_in_meters / converter.width

    disagreement = []
    for frame_idx, (h_before, h_after) in enumerate(zip(before, after)):
        if h_before is None or h_after is None:
            continue
        p_before = cv2.perspectiveTransform(grid, h_before).reshape(-1, 2)
        p_after = cv2.perspectiveTransform(grid, h_after).reshape(-1, 2)
        disagreement.append((np.median(np.linalg.norm(p_before - p_after, axis=1)) * meters_per_pixel, frame_idx))

    # Pick the frames that disagree most, at least half a second apart
    chosen = []
    for meters, frame_idx in sorted(disagreement, reverse=True):
        if all(abs(frame_idx - other) >= 15 for _, other in chosen):
            chosen.append((meters, frame_idx))
        if len(chosen) == count:
            break

    before_drawer = CourtLinesDrawer(line_color=(0, 0, 255))
    after_drawer = CourtLinesDrawer(line_color=(0, 255, 0))
    tactical_view_drawer = TacticalViewDrawer()
    saved = []
    for meters, frame_idx in chosen:
        panels = []
        for drawer, homographies, label in ((before_drawer, before, 'original'), (after_drawer, after, 'improved')):
            homography = homographies[frame_idx]
            panel = next(drawer.draw([frames[frame_idx]], [homography], converter.court_polylines))
            positions = converter.transform_players_to_tactical_view([homography], [player_tracks[frame_idx]])
            panel = next(tactical_view_drawer.draw([panel], converter.court_image_path, converter.width, converter.height,
                                                   converter.key_points, positions, [player_assignment[frame_idx]]))
            cv2.putText(panel, f'{label}  (video {video_idx}, frame {frame_idx})', (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            panels.append(panel)
        path = os.path.join(figure_dir, f'homography_video{video_idx}_frame{frame_idx}.jpg')
        cv2.imwrite(path, np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append((path, meters))
    return saved


def main():
    parser = argparse.ArgumentParser(description='Compare homography settings on the sample videos')
    parser.add_argument('--videos', nargs='+', default=['1', '2', '3'])
    parser.add_argument('--video_dir', default='input_videos')
    parser.add_argument('--stub_root', default='stubs')
    parser.add_argument('--lag', type=int, default=10, help='frames between the two positions used for speed')
    parser.add_argument('--figure_dir', default=None, help='where to save before/after overlays (optional)')
    parser.add_argument('--league', default='nba', choices=['fiba', 'nba'], help='court dimensions (the sample videos are NBA)')
    args = parser.parse_args()

    per_setting = {name: [] for name, *_ in SETTINGS}
    for video_idx in args.videos:
        stub_dir = os.path.join(args.stub_root, f'video_{video_idx}')
        with open(os.path.join(stub_dir, 'court_key_points_stub.pkl'), 'rb') as f:
            raw_keypoints = pickle.load(f)
        with open(os.path.join(stub_dir, 'player_track_stubs.pkl'), 'rb') as f:
            player_tracks = pickle.load(f)
        video_path = os.path.join(args.video_dir, f'video_{video_idx}.mp4')
        fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) or 30
        camera_shots = CameraCutDetector().get_segments(VideoFrames(video_path), read_from_stub=True,
                                                        stub_path=os.path.join(stub_dir, 'camera_cut_stub.pkl'))

        rows = []
        homographies_by_setting = {}
        for name, *setting in SETTINGS:
            converter, homographies = get_homographies(raw_keypoints, args.league, camera_shots, *setting)
            homographies_by_setting[name] = homographies
            measurements = collect_measurements(converter, homographies, raw_keypoints, player_tracks, fps, args.lag)
            per_setting[name].append(measurements)
            rows.append((name, summarize(measurements)))
        print_table(f'video_{video_idx}', rows)

        if args.figure_dir:
            os.makedirs(args.figure_dir, exist_ok=True)
            with open(os.path.join(stub_dir, 'player_assignment_stub.pkl'), 'rb') as f:
                player_assignment = pickle.load(f)
            saved = save_comparison_figures(video_idx, read_video(video_path), player_tracks, player_assignment,
                                            converter, homographies_by_setting['original (conf>=0.5)'],
                                            homographies_by_setting[SETTINGS[-1][0]], args.figure_dir)
            for path, meters in saved:
                print(f'saved {path} (settings disagree by {meters:.1f} m)')

    print_table('all videos', [(name, summarize(merge(per_setting[name]))) for name, *_ in SETTINGS])


if __name__ == '__main__':
    main()
