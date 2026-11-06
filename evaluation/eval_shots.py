"""
Evaluates shot detection and the two/three-point classification against labelled shots.

Run from the repository root after main.py has been run on every labelled video (it creates
the stubs this script reads, including the hoop detections and the camera shots):

    python evaluation/eval_shots.py

evaluation/shot_labels.json lists, per video, the league (court dimensions) and every shot
attempt with the approximate release frame and its value (2 or 3); free throws are not shot
attempts here. For game footage the attempts come from the play-by-play (with its game clock,
the player and whether the shot was made), aligned with the video by
evaluation/align_play_by_play.py, and the release frames were found by looking at the frames.
Shots on which the shooter was fouled and missed are labelled too: the play-by-play does not
count them as field goal attempts, but they are shots, and the number of free throws gives
their value.
A detected shot matches a labelled one if their release frames are at most MATCH_FRAMES apart.

Besides the final settings, the script reruns the detection with each change undone (the ball
filter per camera shot with a shape check and a speed limit in ball sizes, the team check, the
hoop area reaching higher above the rim, leaving out shots without a court position, the court
homography adjusted to the painted lines, the homography checked and smoothed per camera shot,
the improved homography altogether, the position at take-off instead of the release) and with
the team check, the hoop area, the court position and the camera shots undone (the first
version), to show what each change contributes.
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tactical_view_converter import TacticalViewConverter, CourtLineRefiner
from camera_cut_detector import CameraCutDetector
from trackers import BallTracker
from shot_detector import ShotDetector
from utils import VideoFrames

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shot_labels.json')
# Release frames of a detected and a labelled shot may differ by this much (1 s at 30 fps, 1.2 s at 25 fps)
MATCH_FRAMES = 30

FIRST_VERSION = {'use_teams': False, 'hoop_zone_top': -2.0, 'require_court_position': False}
# The ball filter before it ran per camera shot with a shape check and a speed limit in ball sizes
EARLIER_BALL_FILTER = {'maximum_speed_in_pixels': 25, 'ball_aspect_range': None, 'per_camera_shot': False}
# name, homography estimation (see get_homographies), ShotDetector settings that differ from the
# defaults; the key 'ball_filter' holds BallTracker.remove_wrong_detections settings instead
SETTINGS = [
    ('final', 'final', {}),
    ('earlier ball filter', 'final', {'ball_filter': EARLIER_BALL_FILTER}),
    ('first version', 'without camera shots', FIRST_VERSION),
    ('no team check', 'final', {'use_teams': False}),
    ('hoop area up to 2 hoop widths above the rim', 'final', {'hoop_zone_top': -2.0}),
    ('shots without court position kept', 'final', {'require_court_position': False}),
    ('homography without line refinement', 'without line refinement', {}),
    ('homography without camera shots', 'without camera shots', {}),
    ('original homography', 'original', {}),
    ('position at release', 'final', {'use_release_frame': True}),
]


def get_homographies(keypoints, league, estimation, camera_shots, video, stub_dir):
    """
    'final': as in main.py; 'without line refinement': the same without CourtLineRefiner;
    'without camera shots': fits to four keypoints allowed, no limit on their error, smoothed over
    five frames regardless of cuts (the version before the camera shots); 'original': the
    repository's plain least squares on the validated keypoints.
    """
    if estimation in ('final', 'without line refinement'):
        converter = TacticalViewConverter('./images/basketball_court.png', league=league)
        keypoints = converter.filter_low_confidence_keypoints(keypoints)
        if estimation == 'without line refinement':
            return converter, converter.get_homographies(keypoints, camera_shots)
        # main.py's cache of the adjustment (computed from the video if it is missing or stale)
        return converter, CourtLineRefiner(converter).get_homographies(
            VideoFrames(os.path.join('input_videos', video + '.mp4')), keypoints, camera_shots,
            read_from_stub=True, stub_path=os.path.join(stub_dir, 'court_line_refinement_stub.pkl'))
    if estimation == 'without camera shots':
        converter = TacticalViewConverter('./images/basketball_court.png', league=league, min_inliers=4,
                                          max_inlier_error_in_meters=None)
        return converter, converter.get_homographies(converter.filter_low_confidence_keypoints(keypoints))
    converter = TacticalViewConverter('./images/basketball_court.png', league=league, ransac_threshold_in_meters=None,
                                      fix_mirrored_labels=False, smoothing_window=1)
    keypoints = converter.validate_keypoints(converter.filter_low_confidence_keypoints(keypoints))
    return converter, converter.get_homographies(keypoints)


def detect(stub_dir, video, league, estimation, settings):
    def load(name):
        with open(os.path.join(stub_dir, name), 'rb') as f:
            return pickle.load(f)

    # The camera shots come from main.py's cache (computed from the video if it is missing)
    camera_shots = CameraCutDetector().get_segments(VideoFrames(os.path.join('input_videos', video + '.mp4')),
                                                    read_from_stub=True,
                                                    stub_path=os.path.join(stub_dir, 'camera_cut_stub.pkl'))
    converter, homographies = get_homographies(load('court_key_points_stub.pkl'), league, estimation, camera_shots,
                                               video, stub_dir)
    detector = ShotDetector(converter)
    ball_filter = dict(settings.get('ball_filter', {}))
    for name, value in settings.items():
        if name != 'ball_filter':
            setattr(detector, name, value)
    per_camera_shot = ball_filter.pop('per_camera_shot', True)
    ball_tracks = BallTracker.remove_wrong_detections(None, load('ball_track_stubs.pkl'),
                                                      camera_shots=camera_shots if per_camera_shot else None, **ball_filter)
    return detector.detect_shots(ball_tracks, load('hoop_track_stubs.pkl'), load('player_track_stubs.pkl'), homographies,
                                 load('player_assignment_stub.pkl'))


def match(detected, labelled):
    """Greedily pairs detected and labelled shots by release frame. Returns (pairs, unmatched detected, unmatched labelled)."""
    candidates = sorted((abs(d['release_frame'] - l['frame']), i, j) for i, d in enumerate(detected)
                        for j, l in enumerate(labelled) if abs(d['release_frame'] - l['frame']) <= MATCH_FRAMES)
    used_detected, used_labelled, pairs = set(), set(), []
    for _, i, j in candidates:
        if i not in used_detected and j not in used_labelled:
            used_detected.add(i); used_labelled.add(j)
            pairs.append((detected[i], labelled[j]))
    return (pairs, [d for i, d in enumerate(detected) if i not in used_detected],
            [l for j, l in enumerate(labelled) if j not in used_labelled])


def describe(shot):
    """The play-by-play clock, player and note of a labelled shot."""
    made = 'fouled' if shot.get('fouled') else {True: 'made', False: 'missed'}.get(shot.get('made'), '')
    return ', '.join(part for part in (shot.get('clock'), shot.get('player'), made, shot.get('note')) if part)


def margin_band(margin):
    if margin is None:
        return 'no court position'
    distance = abs(margin)
    return '< 0.5 m' if distance < 0.5 else ('0.5-1.5 m' if distance < 1.5 else '> 1.5 m')


def main():
    parser = argparse.ArgumentParser(description='Evaluate shot detection and two/three-point classification')
    parser.add_argument('--labels', default=LABELS_PATH)
    parser.add_argument('--stub_root', default='stubs')
    parser.add_argument('--videos', nargs='+', help='Only these labelled videos (default: all of them)')
    args = parser.parse_args()

    with open(args.labels) as f:
        labels = json.load(f)

    print('| setting | labelled shots | detected | matched | missed | false detections | correct 2/3 (of matched) | 2 called 3 | 3 called 2 |')
    print('|---|---|---|---|---|---|---|---|---|')
    per_shot_rows = []
    for name, estimation, settings in SETTINGS:
        totals = Counter()
        bands = Counter()
        for video, video_info in labels['videos'].items():
            if args.videos and video not in args.videos:
                continue
            labelled = [shot for shot in labels['shots'] if shot['video'] == video]
            detected = detect(os.path.join(args.stub_root, video), video, video_info['league'], estimation, settings)
            pairs, false_detections, missed = match(detected, labelled)
            totals['labelled'] += len(labelled)
            totals['detected'] += len(detected)
            totals['matched'] += len(pairs)
            totals['missed'] += len(missed)
            totals['false'] += len(false_detections)
            for d, l in pairs:
                correct = d['points'] == l['points']
                totals['correct'] += correct
                totals['2 as 3'] += l['points'] == 2 and d['points'] == 3
                totals['3 as 2'] += l['points'] == 3 and d['points'] == 2
                bands[(margin_band(d['three_point_margin']), correct)] += 1
                if name == 'final':
                    margin = d['three_point_margin']
                    per_shot_rows.append(f"| {video} | {l['frame']} | {l['points']} | {d['release_frame']} | {d['points']} | "
                                         f"{'-' if margin is None else f'{margin:+.2f}'} | {describe(l)} |")
            if name == 'final':
                per_shot_rows += [f"| {video} | {l['frame']} | {l['points']} | missed | | | {describe(l)} |" for l in missed]
                per_shot_rows += [f"| {video} | | | {d['release_frame']} | {d['points']} | | false detection |" for d in false_detections]
        accuracy = f"{totals['correct']}/{totals['matched']}" + (f" ({totals['correct'] / totals['matched']:.0%})" if totals['matched'] else '')
        print(f"| {name} | {totals['labelled']} | {totals['detected']} | {totals['matched']} | {totals['missed']} | "
              f"{totals['false']} | {accuracy} | {totals['2 as 3']} | {totals['3 as 2']} |")
        if name == 'final':
            final_bands = bands

    print('\n## final setting: accuracy by distance of the take-off spot from the three-point line')
    print('| distance from the line | correct | wrong |')
    print('|---|---|---|')
    for band in ('< 0.5 m', '0.5-1.5 m', '> 1.5 m', 'no court position'):
        print(f'| {band} | {final_bands[(band, True)]} | {final_bands[(band, False)]} |')

    print('\n## final setting: every shot')
    print('| video | labelled release frame | labelled points | detected release frame | detected points | margin to the line (m) | note |')
    print('|---|---|---|---|---|---|---|')
    print('\n'.join(per_shot_rows))


if __name__ == '__main__':
    main()
