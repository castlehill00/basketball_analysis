"""
Compares letterboxed and stretched input for the ball detector against the ball labels.

Run from the repository root:

    python evaluation/eval_ball_preprocessing.py

The ball model was trained on frames stretched to 640x640. YOLO letterboxes by default (scales
the frame keeping its aspect ratio and pads it), so the model sees a round ball where it learned
a stretched one. The giveaway is the box shape: a ball should get a square box in the frame,
but letterboxed inference returns boxes about 1.6 times taller than wide.

For each setting this runs the detector on every frame, keeps the highest-confidence ball as
the pipeline does, and reports the detections before and after the jump filter. "Frames with
the real ball" counts frames where the kept detection is a labelled real ball.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from configs import BALL_DETECTOR_PATH
from trackers import BallTracker
from utils import read_video
from eval_ball_filter import FILTERS, count, label_of, load_labels

SETTINGS = [('letterbox (original)', False), ('stretch to 640x640', True)]


def main():
    parser = argparse.ArgumentParser(description='Compare letterboxed and stretched input for the ball detector')
    parser.add_argument('--videos', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--video_dir', default='input_videos')
    args = parser.parse_args()

    labels = load_labels()
    trackers = {name: BallTracker(BALL_DETECTOR_PATH, stretch_input=stretch) for name, stretch in SETTINGS}
    detection_counts = {name: Counter() for name, _ in SETTINGS}
    aspect_ratios = {name: [] for name, _ in SETTINGS}
    filter_counts = {(name, filter_name): Counter() for name, _ in SETTINGS for filter_name, _ in FILTERS}
    total_frames = 0

    for video_idx in args.videos:
        frames = read_video(os.path.join(args.video_dir, f'video_{video_idx}.mp4'))
        total_frames += len(frames)
        video_labels = labels.get(video_idx, {})
        for name, _ in SETTINGS:
            ball_positions = trackers[name].get_object_tracks(frames)
            for frame_num, positions in enumerate(ball_positions):
                if not positions.get(1):
                    continue
                bbox = positions[1]['bbox']
                label = label_of(video_labels.get(frame_num, []), bbox)
                detection_counts[name][label] += 1
                if label == 'ball':
                    aspect_ratios[name].append((bbox[3] - bbox[1]) / (bbox[2] - bbox[0]))
            for filter_name, filter_fn in FILTERS:
                filter_counts[(name, filter_name)] += count(filter_fn, ball_positions, video_labels)

    print(f'\n## detections ({total_frames} frames)')
    print('| input | frames with a detection | real ball | not a ball | real-ball share | ball box height/width (median) |')
    print('|---|---|---|---|---|---|')
    for name, _ in SETTINGS:
        c = detection_counts[name]
        detected = sum(c.values())
        print(f"| {name} | {detected} | {c['ball']} | {c['not_ball']} | {c['ball'] / (c['ball'] + c['not_ball']):.0%} | "
              f"{np.median(aspect_ratios[name]):.2f} |")
        if c['unlabelled']:
            print(f"|  | ({c['unlabelled']} not covered by the labels, {c['unclear']} unclear) | | | | |")

    print('\n## after the jump filter')
    print('| input | filter | real balls kept | false detections removed | frames with the real ball | kept that are real balls |')
    print('|---|---|---|---|---|---|')
    for name, _ in SETTINGS:
        for filter_name, _ in FILTERS:
            c = filter_counts[(name, filter_name)]
            balls = c[('ball', True)] + c[('ball', False)]
            false = c[('not_ball', True)] + c[('not_ball', False)]
            kept = c[('ball', True)] + c[('not_ball', True)]
            print(f"| {name} | {filter_name} | {c[('ball', True)]}/{balls} ({c[('ball', True)] / balls:.0%}) | "
                  f"{c[('not_ball', False)]}/{false} ({c[('not_ball', False)] / false:.0%}) | "
                  f"{c[('ball', True)]} ({c[('ball', True)] / total_frames:.0%}) | {c[('ball', True)] / kept:.0%} |")


if __name__ == '__main__':
    main()
