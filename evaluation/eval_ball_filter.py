"""
Compares ball jump filters against manual labels of the ball detections.

Run from the repository root after main.py has produced the stubs of each video:

    python evaluation/eval_ball_filter.py

evaluation/ball_labels.json labels ball detections of the three sample videos as 'ball',
'not_ball' or 'unclear'. It covers the highest-confidence 'Ball' box of every frame from both
letterboxed and stretched inference (see eval_ball_preprocessing.py); a detection takes the
label of the labelled box at the same place. Each box was judged by eye from a crop around it;
'ball' labels that implied an impossible jump, or two balls in one frame, were then rechecked
against the surrounding frames. 'unclear' detections are left out of the counts.
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter
from copy import deepcopy

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from trackers import BallTracker

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ball_labels.json')
# Pixels between box centres for a detection to count as a labelled one
MATCH_DISTANCE = 15


def original_remove_wrong_detections(ball_positions):
    """The repository's original filter, kept here for comparison."""
    maximum_allowed_distance = 25
    last_good_frame_index = -1

    for i in range(len(ball_positions)):
        current_box = ball_positions[i].get(1, {}).get('bbox', [])
        if len(current_box) == 0:
            continue
        if last_good_frame_index == -1:
            last_good_frame_index = i
            continue

        last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
        frame_gap = i - last_good_frame_index
        if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > maximum_allowed_distance * frame_gap:
            ball_positions[i] = {}
        else:
            last_good_frame_index = i

    return ball_positions


FILTERS = [
    ('original (compare with last kept)', original_remove_wrong_detections),
    ('longest plausible chain', lambda positions: BallTracker.remove_wrong_detections(None, positions, rescue_detections_on_path=False)),
    ('+ put back detections on the path (final)', lambda positions: BallTracker.remove_wrong_detections(None, positions)),
]


def load_labels(path=LABELS_PATH):
    """Returns {video: {frame: [(box centre, label), ...]}}."""
    with open(path) as f:
        entries = json.load(f)
    labels = {}
    for entry in entries:
        x1, y1, x2, y2 = entry['bbox']
        labels.setdefault(entry['video'], {}).setdefault(entry['frame'], []).append(
            (np.array([(x1 + x2) / 2, (y1 + y2) / 2]), entry['label']))
    return labels


def label_of(frame_labels, bbox):
    """Label of the labelled box at the same place as bbox, or 'unlabelled'."""
    x1, y1, x2, y2 = bbox
    centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    for labelled_centre, label in frame_labels:
        if np.linalg.norm(centre - labelled_centre) <= MATCH_DISTANCE:
            return label
    return 'unlabelled'


def count(filter_fn, ball_positions, video_labels):
    """Counts (label, kept by the filter) over the frames that have a ball detection."""
    kept = filter_fn(deepcopy(ball_positions))
    counts = Counter()
    for frame_num, positions in enumerate(ball_positions):
        if positions.get(1):
            label = label_of(video_labels.get(frame_num, []), positions[1]['bbox'])
            counts[(label, bool(kept[frame_num].get(1)))] += 1
    return counts


def print_table(title, rows):
    print(f'\n## {title}')
    print('| filter | real balls kept | false detections removed | false detections kept | kept that are real balls |')
    print('|---|---|---|---|---|')
    for name, c in rows:
        balls = c[('ball', True)] + c[('ball', False)]
        false = c[('not_ball', True)] + c[('not_ball', False)]
        kept = c[('ball', True)] + c[('not_ball', True)]
        print(f"| {name} | {c[('ball', True)]}/{balls} ({c[('ball', True)] / balls:.0%}) | "
              f"{c[('not_ball', False)]}/{false} ({c[('not_ball', False)] / false:.0%}) | {c[('not_ball', True)]} | "
              f"{c[('ball', True)] / kept:.0%} |")
    unlabelled = rows[0][1][('unlabelled', True)] + rows[0][1][('unlabelled', False)]
    if unlabelled:
        print(f'\n{unlabelled} detections are not covered by the labels and are left out.')


def main():
    parser = argparse.ArgumentParser(description='Compare ball jump filters against manual labels')
    parser.add_argument('--videos', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--stub_root', default='stubs')
    args = parser.parse_args()

    labels = load_labels()
    totals = {name: Counter() for name, _ in FILTERS}
    for video_idx in args.videos:
        with open(os.path.join(args.stub_root, f'video_{video_idx}', 'ball_track_stubs.pkl'), 'rb') as f:
            ball_positions = pickle.load(f)

        rows = []
        for name, filter_fn in FILTERS:
            counts = count(filter_fn, ball_positions, labels.get(video_idx, {}))
            totals[name] += counts
            rows.append((name, counts))
        print_table(f'video_{video_idx}', rows)

    print_table('all videos', [(name, totals[name]) for name, _ in FILTERS])


if __name__ == '__main__':
    main()
