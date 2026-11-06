"""
Measures how well the ball shown in the output video matches where the ball is, on game footage.

Run from the repository root after main.py has been run on the labelled videos (it creates the
ball and camera shot stubs this script reads):

    python evaluation/eval_ball_frames.py --videos wc23_final_q1 wc23_final_q2

evaluation/ball_frame_labels.json labels randomly chosen frames of each video: 'ball' with the
centre and diameter of the ball (in frame pixels) where a ball can be seen, 'none' where no ball
can be seen (close-ups, crowd shots, replays without the ball, graphics) and 'unclear' where it
could not be told (motion blur, a ball half hidden in a crowd); 'unclear' frames are left out.
The labels were made from enlarged frames with every candidate box of the ball detector drawn
in (down to 0.1 confidence), without looking at the output of any of the settings compared.

A frame counts as
    correct       the ball is shown within one ball diameter (at least 20 px) of the ball
    wrong place   the ball is shown somewhere else
    missed        a ball can be seen but none is shown
    false         a ball is shown although none can be seen

A second table checks the ball holder marked in the video. There are no labels of who has the
ball, so the reference is the player the possession rules (BallAquisitionDetector) pick for the
labelled ball, i.e. the holder they would find with a perfectly detected ball; in frames with no
ball in the camera views other than the main one (evaluation/view_labels.json) nobody should be
marked. It also gives the share of main-camera frames with a holder marked.
"""
import argparse
import json
import os
import pickle
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from trackers import BallTracker
from camera_cut_detector import CameraCutDetector
from ball_aquisition import BallAquisitionDetector

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ball_frame_labels.json')
VIEW_LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'view_labels.json')
MIN_TOLERANCE = 20
MAX_GAP = 6
EARLIER_FILTER = {'maximum_speed_in_pixels': 25, 'ball_aspect_range': None}


def settings(raw, camera_shots):
    """name -> ball tracks as the output video would show them."""
    earlier = BallTracker.remove_wrong_detections(None, deepcopy(raw), **EARLIER_FILTER)
    final = BallTracker.remove_wrong_detections(None, deepcopy(raw), camera_shots=camera_shots)
    return [
        ('detections (confidence >= 0.5)', raw),
        ('original: earlier filter, every gap filled', BallTracker.interpolate_ball_positions(None, deepcopy(earlier))),
        (f'earlier filter, gaps <= {MAX_GAP} frames within camera shots',
         BallTracker.interpolate_ball_positions(None, deepcopy(earlier), camera_shots, MAX_GAP)),
        ('final filter, no gaps filled', final),
        (f'final filter, gaps <= {MAX_GAP} frames within camera shots (final)',
         BallTracker.interpolate_ball_positions(None, deepcopy(final), camera_shots, MAX_GAP)),
    ]


def holder_settings(shown_tracks, player_tracks, camera_shots):
    """name -> ball holder per frame, from the ball tracks of settings()."""
    shown = dict(shown_tracks)
    detector = BallAquisitionDetector()
    return [
        ('original: original ball, original possession',
         detector.detect_ball_possession(player_tracks, shown['original: earlier filter, every gap filled'])),
        ('final ball, original possession',
         detector.detect_ball_possession(player_tracks, shown[f'final filter, gaps <= {MAX_GAP} frames within camera shots (final)'])),
        ('final ball, possession carried over gaps (final)',
         detector.detect_ball_possession(player_tracks, shown[f'final filter, gaps <= {MAX_GAP} frames within camera shots (final)'],
                                         camera_shots)),
    ]


def score_holder(possession, frame_labels, player_tracks, views):
    """Counts of the holder marked in the labelled frames (see the module docstring)."""
    detector = BallAquisitionDetector()
    counts = {'correct': 0, 'wrong': 0, 'missed': 0, 'with holder': 0, 'false': 0, 'no ball': 0}
    for frame, label in frame_labels.items():
        frame = int(frame)
        holder = possession[frame]
        if label['label'] == 'ball':
            x, y, half = label['center'][0], label['center'][1], label['size'] / 2
            reference = detector.find_best_candidate_for_possession(
                (x, y), player_tracks[frame], [x - half, y - half, x + half, y + half])
            if reference == -1:
                counts['false'] += holder != -1
                continue
            counts['with holder'] += 1
            counts['correct' if holder == reference else ('missed' if holder == -1 else 'wrong')] += 1
        elif label['label'] == 'none' and views[frame] == 0:
            counts['no ball'] += 1
            counts['false'] += holder != -1
    return counts


def frame_views(segments, length):
    """1 = main camera, 0 = another view, -1 = main camera zoomed in."""
    views = np.full(length, -1)
    for first, last, kind in segments:
        views[first:last + 1] = {'main': 1, 'other': 0, 'zoomed': -1}[kind]
    return views


def score(ball_tracks, frame_labels):
    counts = {'correct': 0, 'wrong place': 0, 'missed': 0, 'false': 0, 'no ball': 0, 'ball': 0}
    for frame, label in frame_labels.items():
        if label['label'] == 'unclear':
            continue
        bbox = ball_tracks[int(frame)].get(1, {}).get('bbox', [])
        shown = len(bbox) == 4 and not np.any(np.isnan(bbox))
        if label['label'] == 'none':
            counts['no ball'] += 1
            counts['false'] += shown
            continue
        counts['ball'] += 1
        if not shown:
            counts['missed'] += 1
            continue
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        distance = np.linalg.norm(center - np.array(label['center']))
        counts['correct' if distance <= max(label['size'], MIN_TOLERANCE) else 'wrong place'] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description='Compare the shown ball with labelled frames')
    parser.add_argument('--videos', nargs='+', default=None, help='Labelled videos (default: all)')
    parser.add_argument('--stub_root', default='stubs')
    args = parser.parse_args()

    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(VIEW_LABELS_PATH) as f:
        view_labels = json.load(f)
    totals, holder_totals = {}, {}
    for video in args.videos or list(labels):
        stub_dir = os.path.join(args.stub_root, video)
        with open(os.path.join(stub_dir, 'ball_track_stubs.pkl'), 'rb') as f:
            raw = pickle.load(f)
        with open(os.path.join(stub_dir, 'player_track_stubs.pkl'), 'rb') as f:
            player_tracks = pickle.load(f)
        with open(os.path.join(stub_dir, 'camera_cut_stub.pkl'), 'rb') as f:
            camera_shots = CameraCutDetector().find_segments(pickle.load(f))
        views = frame_views(view_labels[video]['segments'], len(raw))
        rows, holder_rows = [], []
        shown_tracks = settings(raw, camera_shots)
        for name, tracks in shown_tracks:
            counts = score(tracks, labels[video]['frames'])
            rows.append((name, counts))
            add(totals, name, counts)
        for name, possession in holder_settings(shown_tracks, player_tracks, camera_shots):
            counts = score_holder(possession, labels[video]['frames'], player_tracks, views)
            possession = np.array(possession)
            counts['main frames'], counts['main frames with holder'] = int(np.sum(views == 1)), int(np.sum(possession[views == 1] != -1))
            holder_rows.append((name, counts))
            add(holder_totals, name, counts)
        print_table(video, rows)
        print_holder_table(video, holder_rows)
    if len(totals) and len(args.videos or labels) > 1:
        print_table('all videos', list(totals.items()))
        print_holder_table('all videos', list(holder_totals.items()))


def add(totals, name, counts):
    total = totals.setdefault(name, dict.fromkeys(counts, 0))
    for key, value in counts.items():
        total[key] += value


def print_holder_table(title, rows):
    print(f'\n## {title}: ball holder')
    print('| setting | main-camera frames with a holder | reference holder: marked | other player marked | none marked | '
          'holder marked without one |')
    print('|---|---|---|---|---|---|')
    for name, c in rows:
        print(f"| {name} | {c['main frames with holder'] / c['main frames']:.1%} | {c['correct']}/{c['with holder']} | "
              f"{c['wrong']} | {c['missed']} | {c['false']} |")


def print_table(title, rows):
    print(f'\n## {title}')
    print('| setting | ball seen: correct | wrong place | missed | no ball: shown anyway | shown balls that are right |')
    print('|---|---|---|---|---|---|')
    for name, c in rows:
        shown = c['correct'] + c['wrong place'] + c['false']
        print(f"| {name} | {c['correct']}/{c['ball']} | {c['wrong place']} | {c['missed']} | {c['false']}/{c['no ball']} | "
              f"{c['correct'] / shown:.0%} |")


if __name__ == '__main__':
    main()
