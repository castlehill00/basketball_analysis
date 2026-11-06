"""
Compares court homography settings on game footage whose camera shots were labelled from their frames.

Run from the repository root after main.py has been run on the labelled videos (it creates the
keypoint and camera shot stubs this script reads):

    python evaluation/eval_court.py --videos wc23_final_q1

evaluation/view_labels.json gives every camera shot of a labelled video as 'main' (the main
broadcast camera), 'other' (close-ups, the cameras behind the baskets or low at the side, the
camera above the court, graphics) or 'zoomed' (the main camera zoomed in on a stoppage; left out
of both rates). The settings are compared on:
    main view          share of main-camera frames that get a homography
    other views        share of other-view frames that get one; it is wrong there, since the
                       keypoint model was only trained on the main camera
    line match         on every 5th main-camera frame, the share of court line points projected
                       into the frame by the homography that land within 4 px of a painted line
                       found in the frame. It does not use the keypoints, and painted lines are
                       often hidden or not found, so only the comparison between settings means
                       something: shifting the court by 0.25 m lowers the median from 0.24 to
                       0.19 on wc23_final_q1, by 0.5 m to 0.13
    poor match         share of those frames with a line match below 0.1
    jitter             second difference over consecutive main-camera frames of where the
                       homography puts court points in the image (median over the points in the
                       frame); a smoothly moving camera gives close to 0
    jumps              how often per minute of main-camera footage the drawn court jumps: the
                       court point in the frame whose position changes most has a second
                       difference above 40 px (runs of such frames within a camera shot count once)
The line match is given on the frames each setting has a homography for (what the output video
shows) and on the frames all settings have one for (the same frames for every setting). The last
setting adjusts the homographies to the same painted lines, so its line match is optimistic;
evaluation/eval_line_refinement.py measures it on lines left out of the adjustment instead.
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tactical_view_converter import TacticalViewConverter, CourtLineRefiner
from tactical_view_converter.court_line_refiner import line_mask
from camera_cut_detector import CameraCutDetector
from utils import VideoFrames

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'view_labels.json')
LINE_FRAME_STEP = 5
MATCH_PIXELS = 4
POOR_MATCH = 0.1
JUMP_PIXELS = 40
# The score bug covers the court below this row (1280x720 frames)
SCORE_BUG_TOP = 610

# name, TacticalViewConverter arguments, keypoints validated as in the original code, use camera
# shots, adjust to the painted lines with CourtLineRefiner
SETTINGS = [
    ('original', {'ransac_threshold_in_meters': None, 'fix_mirrored_labels': False, 'smoothing_window': 1}, True, False, False),
    ('before camera shots', {'min_inliers': 4, 'max_inlier_error_in_meters': None}, False, False, False),
    ('+ stricter frame checks', {}, False, False, False),
    ('+ camera shots, old frame checks', {'min_inliers': 4, 'max_inlier_error_in_meters': None}, False, True, False),
    ('+ both', {}, False, True, False),
    ('+ line refinement (final)', {}, False, True, True),
]


def frame_labels(segments, length):
    """1 = main camera, 0 = other view, -1 = main camera zoomed in."""
    labels = np.full(length, -1)
    for first, last, kind in segments:
        labels[first:last + 1] = {'main': 1, 'other': 0, 'zoomed': -1}[kind]
    return labels


def court_line_points(converter, step_in_meters=0.25):
    """Points every step_in_meters along all court markings, in tactical view pixels."""
    step = converter.meters_to_pixels(step_in_meters)
    points = []
    for polyline in converter.court_polylines:
        polyline = np.asarray(polyline, np.float32)
        for start, end in zip(polyline, polyline[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / step)))
            points.append(start + (end - start) * np.linspace(0, 1, count, endpoint=False)[:, None])
    return np.vstack(points).astype(np.float32)


def line_match(distance_to_line, homography, line_points):
    """
    Share of the projected court line points in view that lie within MATCH_PIXELS of a painted
    line (nan if the homography cannot be inverted: the original estimation sometimes gives a
    singular one).
    """
    try:
        court_to_frame = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return np.nan
    projected = cv2.perspectiveTransform(line_points.reshape(-1, 1, 2), court_to_frame).reshape(-1, 2)
    height, width = distance_to_line.shape
    with np.errstate(invalid='ignore'):
        in_view = np.isfinite(projected).all(axis=1) & (projected[:, 0] >= 0) & (projected[:, 0] < width - 1) & \
            (projected[:, 1] >= 0) & (projected[:, 1] < SCORE_BUG_TOP)
    if in_view.sum() < 20:
        return np.nan
    pixels = np.round(projected[in_view]).astype(int)
    return float(np.mean(distance_to_line[pixels[:, 1], pixels[:, 0]] <= MATCH_PIXELS))


def second_differences(positions, t, converter, frame_width, frame_height):
    """Second difference at frame t of the image positions of the court points in the frame (None if a frame lacks a homography)."""
    if positions[t - 1] is None or positions[t] is None or positions[t + 1] is None:
        return None
    before, now, after = positions[t - 1], positions[t], positions[t + 1]
    use = converter.near_frame(now, frame_width, frame_height, margin=0.0) & \
        np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
    return np.linalg.norm(before[use] - 2 * now[use] + after[use], axis=1)


def jitter(positions, labels, converter, frame_width, frame_height):
    values = []
    for t in range(1, len(positions) - 1):
        differences = second_differences(positions, t, converter, frame_width, frame_height) if labels[t] == 1 else None
        if differences is not None and len(differences):
            values.append(np.median(differences))
    return np.array(values)


def jumps(positions, labels, camera_shots, converter, frame_width, frame_height):
    """Number of jumps of the drawn court in main-camera frames (see the module docstring)."""
    shot_of = np.full(len(positions), -1)
    for index, (first, last) in enumerate(camera_shots):
        shot_of[first:last + 1] = index
    jumping = np.zeros(len(positions), dtype=bool)
    for t in range(1, len(positions) - 1):
        if labels[t] != 1 or not shot_of[t - 1] == shot_of[t] == shot_of[t + 1]:
            continue
        differences = second_differences(positions, t, converter, frame_width, frame_height)
        jumping[t] = differences is not None and len(differences) >= 4 and differences.max() > JUMP_PIXELS
    return int(jumping[0] + np.sum(jumping[1:] & ~jumping[:-1]))


def main():
    parser = argparse.ArgumentParser(description='Compare court homography settings on labelled camera shots')
    parser.add_argument('--videos', nargs='+', default=None, help='Labelled videos (default: all)')
    parser.add_argument('--league', default='fiba')
    parser.add_argument('--stub_root', default='stubs')
    args = parser.parse_args()

    with open(LABELS_PATH) as f:
        view_labels = json.load(f)
    rows = {name: {'main': [], 'other': [], 'match': [], 'jitter': [], 'jumps': 0} for name, *_ in SETTINGS}
    main_minutes = 0.0
    for video in args.videos or list(view_labels):
        stub_dir = os.path.join(args.stub_root, video)
        video_path = os.path.join('input_videos', video + '.mp4')
        with open(os.path.join(stub_dir, 'court_key_points_stub.pkl'), 'rb') as f:
            keypoints = pickle.load(f)
        frame_height, frame_width = keypoints[0].orig_shape
        labels = frame_labels(view_labels[video]['segments'], len(keypoints))
        with open(os.path.join(stub_dir, 'camera_cut_stub.pkl'), 'rb') as f:
            camera_shots = CameraCutDetector().find_segments(pickle.load(f))
        capture = cv2.VideoCapture(video_path)
        main_minutes += np.sum(labels == 1) / capture.get(cv2.CAP_PROP_FPS) / 60
        capture.release()

        homography_sets = {}
        for name, converter_args, validate, use_camera_shots, refine in SETTINGS:
            converter = TacticalViewConverter('./images/basketball_court.png', league=args.league, **converter_args)
            frame_keypoints = converter.filter_low_confidence_keypoints(keypoints)
            if validate:
                frame_keypoints = converter.validate_keypoints(frame_keypoints)
            if refine:
                # main.py's cache of the adjustment (computed from the video if it is missing or stale)
                homographies = CourtLineRefiner(converter).get_homographies(
                    VideoFrames(video_path), frame_keypoints, camera_shots if use_camera_shots else None,
                    read_from_stub=True, stub_path=os.path.join(stub_dir, 'court_line_refinement_stub.pkl'))
            else:
                homographies = converter.get_homographies(frame_keypoints, camera_shots if use_camera_shots else None)
            homography_sets[name] = homographies
            has = np.array([h is not None for h in homographies])
            rows[name]['main'].append(has[labels == 1])
            rows[name]['other'].append(has[labels == 0])
            positions = [None if h is None else converter.project_anchor_points(h, frame_width, frame_height)
                         for h in homographies]
            rows[name]['jitter'].append(jitter(positions, labels, converter, frame_width, frame_height))
            rows[name]['jumps'] += jumps(positions, labels, camera_shots, converter, frame_width, frame_height)
            print(f'{video}: {name} done', flush=True)

        # Line match: decode the video once, find the painted lines in every LINE_FRAME_STEP-th main-camera frame
        line_points = court_line_points(converter)
        wanted = set(t for t in range(0, len(labels), LINE_FRAME_STEP) if labels[t] == 1)
        matches = {name: [] for name in homography_sets}
        capture = cv2.VideoCapture(video_path)
        frame_num = 0
        while frame_num <= max(wanted) and capture.grab():
            if frame_num in wanted:
                distance_to_line = cv2.distanceTransform(1 - line_mask(capture.retrieve()[1])[0], cv2.DIST_L2, 3)
                for name, homographies in homography_sets.items():
                    homography = homographies[frame_num]
                    matches[name].append(np.nan if homography is None else line_match(distance_to_line, homography, line_points))
            frame_num += 1
        capture.release()
        for name in homography_sets:
            rows[name]['match'].append(np.array(matches[name], dtype=float))

    all_matches = {name: np.concatenate(rows[name]['match']) for name, *_ in SETTINGS}
    common = np.all([~np.isnan(match) for match in all_matches.values()], axis=0)

    def describe(match):
        return f'{np.median(match):.3f} | {np.percentile(match, 10):.3f} | {np.mean(match < POOR_MATCH):.1%}'

    print()
    print(f'Line match on the frames each setting has a homography for, and on the {common.sum()} '
          f'frames all settings have one for:')
    print('| setting | main view | other views | line match median | 10th percentile | poor match | '
          'on common frames: median | 10th percentile | poor match | jitter p50 (px) | jitter p99 (px) | '
          f'jumps per minute (> {JUMP_PIXELS} px) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for name, *_ in SETTINGS:
        main_view = np.concatenate(rows[name]['main'])
        other_views = np.concatenate(rows[name]['other'])
        match = all_matches[name]
        shake = np.concatenate(rows[name]['jitter'])
        print(f'| {name} | {main_view.mean():.1%} | {other_views.mean():.1%} ({other_views.sum()} frames) | '
              f'{describe(match[~np.isnan(match)])} | {describe(match[common])} | '
              f'{np.median(shake):.2f} | {np.percentile(shake, 99):.1f} | {rows[name]["jumps"] / main_minutes:.1f} |')


if __name__ == '__main__':
    main()
