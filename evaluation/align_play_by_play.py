"""
Aligns the play-by-play of a quarter with its video through the game clock read from the score
bug (evaluation/game_clock.py), as the first step of labelling the shot attempts.

The play-by-play gives the game clock of every field goal attempt, as recorded by the
statisticians (often a second or two after the release). For every attempt this prints the
frames during which the clock was running from SECONDS_BEFORE seconds before that time to
SECONDS_AFTER seconds after it; the release frame is then found by looking at those frames.
Free throws are left out.

    python evaluation/align_play_by_play.py evaluation/espn_pbp/wc23_final_q1.txt stubs/wc23_final_q1/game_clock.json
"""
import argparse
import json
import re

SECONDS_BEFORE = 5
SECONDS_AFTER = 1
FIELD_GOALS = {'+2 Points': (2, True), '+3 Points': (3, True), 'Missed FG': (2, False), 'Missed 3PT': (3, False)}


def read_field_goals(path):
    """Returns the field goal attempts of a play-by-play file (lines of 'type | M:SS - period | description')."""
    attempts = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            kind, time, description = (part.strip() for part in line.split('|'))
            if kind not in FIELD_GOALS:
                continue
            minutes, seconds = re.match(r'(\d+):(\d+)', time).groups()
            points, made = FIELD_GOALS[kind]
            attempts.append({'clock': int(minutes) * 60 + int(seconds), 'points': points, 'made': made,
                             'player': re.split(r' made | missed ', description)[0], 'description': description})
    return attempts


def shown(seconds):
    """The whole seconds the play-by-play gives for a clock reading (in the last minute the clock shows tenths)."""
    return int(seconds)


def running_frames(clock, fps, low, high):
    """Frames whose clock is between low and high (inclusive, whole seconds) and running or stopped for at most a second."""
    frames = []
    for i, (frame, seconds) in enumerate(clock):
        if seconds is None or not low <= shown(seconds) <= high:
            continue
        # Readings of the same value over more than a second mean the clock is stopped (a dead ball)
        same = [f for f, s in clock[max(0, i - int(fps)):i + 1] if s == seconds]
        if frame - same[0] <= fps:
            frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser(description='Find the frames of every field goal attempt in a play-by-play')
    parser.add_argument('play_by_play')
    parser.add_argument('game_clock')
    args = parser.parse_args()

    with open(args.game_clock) as f:
        data = json.load(f)
    clock, fps = data['clock'], data['fps']

    print('| clock | player | points | made | frames with the clock running nearby | first frame showing the time |')
    print('|---|---|---|---|---|---|')
    for attempt in read_field_goals(args.play_by_play):
        t = attempt['clock']
        frames = running_frames(clock, fps, t - SECONDS_AFTER, t + SECONDS_BEFORE)
        showing = [frame for frame, seconds in clock if seconds is not None and shown(seconds) == t]
        print(f"| {t // 60}:{t % 60:02d} | {attempt['player']} | {attempt['points']} | {'yes' if attempt['made'] else 'no'} | "
              f"{f'{frames[0]}-{frames[-1]}' if frames else '-'} | {showing[0] if showing else '-'} |")


if __name__ == '__main__':
    main()
