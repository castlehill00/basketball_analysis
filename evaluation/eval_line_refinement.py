"""
Checks the court homography adjustment to the painted lines (CourtLineRefiner) on lines it did
not use.

The adjustment makes the projected court lines lie on the painted lines found in the frame, so
measuring how many projected line points land on painted lines afterwards would only show that
the fit worked. Here each group of court lines is left out in turn: the homography is adjusted
to the other lines only, and the share of the left-out group's points within 4 px of a painted
line is compared before and after. If the adjustment corrects the geometry (and does not just
bend the court onto whatever lines are near), the left-out lines get closer too.

Run from the repository root after main.py has created the keypoint stubs:

    python evaluation/eval_line_refinement.py --videos wc23_final_q1 wc23_final_q2

Every STEP-th main-camera frame of evaluation/view_labels.json that has a homography is used,
starting, as main.py does, from the homography fitted to the keypoints and checked and smoothed
within the camera shot.
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
from tactical_view_converter.court_line_refiner import line_mask, line_support, refine_frame, sample_polylines
from camera_cut_detector import CameraCutDetector

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'view_labels.json')
# Indices into CourtDimensions.lines()
LINE_GROUPS = [
    ('sidelines and baselines', [0]),
    ('half-court line and centre circle', [1, 2]),
    ('lanes and free-throw circles', [3, 4, 7, 8]),
    ('three-point lines', [5, 6]),
]
STEP = 25
CHANGE = 0.02
# Besides the usual 4 px, a loose tolerance shows whether large errors (the half-court line 100 px
# off) shrink even where the lines do not end up within a few pixels
LOOSE_PIXELS = 16


def main():
    parser = argparse.ArgumentParser(description='Check the line adjustment of the court homography on left-out lines')
    parser.add_argument('--videos', nargs='+', default=None, help='Labelled videos (default: all)')
    parser.add_argument('--league', default='fiba')
    parser.add_argument('--stub_root', default='stubs')
    parser.add_argument('--step', type=int, default=STEP)
    args = parser.parse_args()

    with open(LABELS_PATH) as f:
        view_labels = json.load(f)
    converter = TacticalViewConverter('./images/basketball_court.png', league=args.league)
    refiner = CourtLineRefiner(converter)
    settings = refiner.settings()
    court_lines = converter.court.lines()
    tactical_to_meters = np.diag(settings['meters_per_tactical_pixel'] + [1.0])
    results = {name: [] for name, _ in LINE_GROUPS}
    results['all lines (not left out)'] = []

    for video in args.videos or list(view_labels):
        with open(os.path.join(args.stub_root, video, 'court_key_points_stub.pkl'), 'rb') as f:
            keypoints = converter.filter_low_confidence_keypoints(pickle.load(f))
        with open(os.path.join(args.stub_root, video, 'camera_cut_stub.pkl'), 'rb') as f:
            camera_shots = CameraCutDetector().find_segments(pickle.load(f))
        checked = converter.get_homographies(keypoints, camera_shots)
        main_frames = set()
        for first, last, kind in view_labels[video]['segments']:
            if kind == 'main':
                main_frames.update(range(first, last + 1))
        wanted = {frame_num: checked[frame_num] for frame_num in sorted(main_frames)[::args.step]
                  if checked[frame_num] is not None}

        capture = cv2.VideoCapture(os.path.join('input_videos', video + '.mp4'))
        frame_num = 0
        while wanted and frame_num <= max(wanted) and capture.grab():
            if frame_num in wanted:
                frame = capture.retrieve()[1]
                height = frame.shape[0]
                top = int(round(height * (1 - settings['bottom_margin'])))
                match = settings['match_pixels'] * height / 720
                distance_to_line = cv2.distanceTransform(1 - line_mask(frame, settings['bottom_margin'])[0], cv2.DIST_L2, 3)
                xy = keypoints[frame_num].xy
                xy = (xy.cpu().numpy() if hasattr(xy, 'cpu') else np.asarray(xy))[0]

                def support(frame_to_tactical, points, pixels):
                    court_to_frame = np.linalg.inv(tactical_to_meters @ frame_to_tactical)
                    return line_support(distance_to_line, court_to_frame / court_to_frame[2, 2], points, top, pixels)

                for name, indices in LINE_GROUPS + [('all lines (not left out)', [])]:
                    left_out = set(indices)
                    used = [line for i, line in enumerate(court_lines) if i not in left_out]
                    checked = [line for i, line in enumerate(court_lines) if i in left_out] or court_lines
                    group_settings = dict(settings, line_points=sample_polylines(used, refiner.line_step_in_meters).tolist())
                    refined, _ = refine_frame(frame, wanted[frame_num], xy, group_settings)
                    if refined is None:
                        continue
                    points = sample_polylines(checked, 0.25)
                    loose = LOOSE_PIXELS * height / 720
                    values = (support(wanted[frame_num], points, match), support(refined, points, match),
                              support(wanted[frame_num], points, loose), support(refined, points, loose))
                    if not np.any(np.isnan(values)):
                        results[name].append(values)
            frame_num += 1
        capture.release()
        print(f'{video}: {len(wanted)} frames done', flush=True)

    print()
    print(f'Share of points of the checked lines within 4 px / {LOOSE_PIXELS} px of a painted line (median over frames)')
    print(f'| lines checked | frames | 4 px before | 4 px after | improved | worse | {LOOSE_PIXELS} px before | {LOOSE_PIXELS} px after | improved | worse |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for name, values in results.items():
        values = np.array(values)
        if len(values) == 0:
            continue
        cells = []
        for before, after in ((values[:, 0], values[:, 1]), (values[:, 2], values[:, 3])):
            cells.append(f'{np.median(before):.3f} | {np.median(after):.3f} | '
                         f'{np.mean(after > before + CHANGE):.0%} | {np.mean(after < before - CHANGE):.0%}')
        print(f'| {name} | {len(values)} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
