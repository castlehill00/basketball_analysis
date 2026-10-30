"""
Summarises the ball, possession and team stages of the pipeline from a video's stubs.

Run from the repository root after main.py has produced the stubs:

    python evaluation/pipeline_stats.py input_videos/video_1.mp4 stubs/video_1

The court homography has its own comparison in eval_homography.py.
"""
import os
import sys
import pickle
from copy import deepcopy

from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import read_video
from trackers import BallTracker
from ball_aquisition import BallAquisitionDetector
from configs import BALL_DETECTOR_PATH


def load(stub_dir, name):
    with open(os.path.join(stub_dir, name), 'rb') as f:
        return pickle.load(f)


def main(video_path, stub_dir):
    player_tracks = load(stub_dir, 'player_track_stubs.pkl')
    raw_ball = load(stub_dir, 'ball_track_stubs.pkl')
    assignment = load(stub_dir, 'player_assignment_stub.pkl')
    n = len(raw_ball)

    raw_hits = sum(1 for b in raw_ball if b.get(1))
    filtered = BallTracker.remove_wrong_detections(None, deepcopy(raw_ball))
    kept = sum(1 for b in filtered if b.get(1))
    ball = BallTracker.interpolate_ball_positions(None, filtered)

    possession = BallAquisitionDetector().detect_ball_possession(player_tracks, ball)
    assigned = sum(1 for p in possession if p != -1)

    teams_seen = {}
    for frame in assignment:
        for player_id, team in frame.items():
            teams_seen.setdefault(player_id, set()).add(team)
    flipped = sum(1 for teams in teams_seen.values() if len(teams) > 1)

    # The detector also has a Hoop class that the pipeline ignores; check how often it fires
    frames = read_video(video_path)
    model = YOLO(BALL_DETECTOR_PATH)
    hoop_id = {v: k for k, v in model.names.items()}['Hoop']
    hoop_frames = 0
    for i in range(0, len(frames), 20):
        for result in model.predict(frames[i:i + 20], conf=0.5, verbose=False):
            hoop_frames += int((result.boxes.cls == hoop_id).any())

    print(f'== {video_path}  ({n} frames)')
    print(f'ball detected (raw)          : {raw_hits}/{n} = {raw_hits / n:.0%}')
    print(f'ball kept after jump filter  : {kept}/{n} = {kept / n:.0%}  ({raw_hits - kept} detections dropped)')
    print(f'frames with possession       : {assigned}/{n} = {assigned / n:.0%}')
    print(f'track ids whose team flipped : {flipped}/{len(teams_seen)}')
    print(f'hoop detected                : {hoop_frames}/{len(frames)} = {hoop_frames / len(frames):.0%}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
