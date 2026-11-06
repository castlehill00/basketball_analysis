"""
Reads the game clock from the score bug of the 2023 FIBA World Cup final broadcast, so that the
play-by-play (which gives the game clock of every shot) can be aligned with video frames.

The clock is white text on blue between the period and the shot clock: M:SS, or SS.t in the
last minute. Each character is compared with digit templates cut from frames whose clock was
read by eye. Create the templates once, then read a video:

    python evaluation/game_clock.py templates input_videos/wc23_final_q1.mp4 200=9:57 500=9:45 ...
    python evaluation/game_clock.py read input_videos/wc23_final_q1.mp4 stubs/wc23_final_q1/game_clock.json

The reading is None when the score bug is not shown (replays, close-ups) or a character does
not look like any digit.
"""
import argparse
import json
import os

import cv2
import numpy as np

TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clock_digits.png')
# Clock area of a 1280x720 frame, between the divider after the period and the one before the shot clock
CLOCK_AREA = (903, 632, 984, 662)
# The divider after the period; it is only there while the score bug is shown
DIVIDER_X = 900
# While a sponsor's logo is shown in place of the left score, the whole score bug is moved right by 120 pixels
LAYOUT_SHIFTS = (0, 120)
WHITE = 150
# Every character is centred on a canvas of this size (height, width), so narrow digits stay narrow
CANVAS = (22, 14)
# A character is a digit only if it is at least this close to its template (mean squared difference)
MAX_DIGIT_DISTANCE = 0.03


def find_layout_shift(frame):
    """Returns how far the score bug is moved right (one of LAYOUT_SHIFTS), or None if it is not shown."""
    y1, y2 = CLOCK_AREA[1], CLOCK_AREA[3]
    for shift in LAYOUT_SHIFTS:
        # The divider is a white line, one or two pixels wide, with no white on either side
        columns = (frame[y1:y2, DIVIDER_X + shift - 4:DIVIDER_X + shift + 6].min(axis=2) > WHITE).mean(axis=0)
        if any(columns[i] > 0.8 and columns[i - 3] < 0.5 and columns[i + 3] < 0.5 for i in range(3, 7)):
            return shift
    return None


def find_characters(frame, shift):
    """Returns the digit images and the punctuation marks ('colon' or 'point') of the clock, left to right."""
    x1, y1, x2, y2 = CLOCK_AREA
    x1, x2 = x1 + shift, x2 + shift
    gray = frame[y1:y2, x1:x2].min(axis=2).astype(np.float32) / 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats((gray > WHITE / 255).astype(np.uint8), connectivity=8)
    characters = []
    for label in sorted(range(1, count), key=lambda label: tuple(stats[label])):
        x, y, w, h, area = stats[label]
        if h >= 14 and w <= CANVAS[1] and h <= CANVAS[0]:
            # The digit with its blurred edge, on black (the blue behind it varies)
            mask = cv2.dilate((labels[y:y + h, x:x + w] == label).astype(np.uint8), np.ones((3, 3), np.uint8))
            canvas = np.zeros(CANVAS, np.float32)
            top, left = (CANVAS[0] - h) // 2, (CANVAS[1] - w) // 2
            canvas[top:top + h, left:left + w] = gray[y:y + h, x:x + w] * mask
            characters.append(('digit', x, canvas))
        elif 6 <= area <= 30 and h <= 7 and w <= 7:
            # A colon is two dots above each other, a decimal point one dot at the bottom
            if characters and characters[-1][0] == 'dot' and abs(characters[-1][1] - x) <= 2:
                characters[-1] = ('colon', x, None)
            else:
                characters.append(('dot', x, None))
        elif area >= 6:
            return None  # Something else is drawn here
    return [('point' if kind == 'dot' else kind, x, image) for kind, x, image in characters]


def load_templates(path=TEMPLATES_PATH):
    strip = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    return [strip[:, d * CANVAS[1]:(d + 1) * CANVAS[1]] for d in range(10)]


def shifted(image):
    """The image moved by up to one pixel in each direction (characters are not always centred alike)."""
    padded = np.pad(image, 1)
    return [padded[1 + dy:CANVAS[0] + 1 + dy, 1 + dx:CANVAS[1] + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def distance(image, template):
    """Mean squared difference at the best shift."""
    return min(np.mean((moved - template) ** 2) for moved in shifted(image))


def read_clock(frame, templates):
    """Returns the seconds left on the game clock, or None if it cannot be read."""
    shift = find_layout_shift(frame)
    if shift is None:
        return None
    characters = find_characters(frame, shift)
    if not characters:
        return None
    kinds = [kind for kind, _, _ in characters]
    separators = [i for i, kind in enumerate(kinds) if kind != 'digit']
    if len(separators) != 1:
        return None
    digits = []
    for kind, _, image in characters:
        if kind == 'digit':
            distances = [distance(image, template) for template in templates]
            if min(distances) > MAX_DIGIT_DISTANCE:
                return None
            digits.append(int(np.argmin(distances)))
    split = separators[0]
    if split == 0:
        return None
    left = int(''.join(map(str, digits[:split])))
    right = digits[split:]
    if kinds[split] == 'colon' and len(right) == 2 and 0 <= left <= 10 and right[0] <= 5:
        return left * 60 + right[0] * 10 + right[1]
    if kinds[split] == 'point' and len(right) == 1 and 0 <= left < 60:
        return left + right[0] / 10
    return None


def read_frames(video_path, frame_numbers):
    """Yields (frame number, frame) for the wanted frames, decoding the video in order."""
    capture = cv2.VideoCapture(video_path)
    wanted = sorted(frame_numbers)
    index = 0
    while wanted and capture.grab():
        if index == wanted[0]:
            wanted.pop(0)
            yield index, capture.retrieve()[1]
        index += 1
    capture.release()


def make_templates(video_path, readings, path=TEMPLATES_PATH):
    """Averages every digit of the given frames (frame number -> clock as shown, e.g. '9:57' or '41.7')."""
    samples = [[] for _ in range(10)]
    for index, frame in read_frames(video_path, readings):
        shift = find_layout_shift(frame)
        if shift is None:
            raise ValueError(f'Frame {index}: no score bug')
        characters = find_characters(frame, shift)
        digit_images = [image for kind, _, image in characters if kind == 'digit']
        text = [int(c) for c in readings[index] if c.isdigit()]
        if len(digit_images) != len(text):
            raise ValueError(f'Frame {index}: found {len(digit_images)} digits, expected {readings[index]}')
        for digit, image in zip(text, digit_images):
            samples[digit].append(image)
    missing = [d for d in range(10) if not samples[d]]
    if missing:
        raise ValueError(f'No sample of the digits {missing}')
    templates = []
    for images in samples:
        # Align the samples with their mean before averaging, so the template is not blurred by their offsets
        template = np.mean(images, axis=0)
        for _ in range(3):
            template = np.mean([min(shifted(image), key=lambda moved: np.mean((moved - template) ** 2))
                                for image in images], axis=0)
        templates.append(template)
    strip = np.hstack(templates)
    cv2.imwrite(path, np.round(strip * 255).astype(np.uint8))
    print(f'Saved {path} ({", ".join(f"{d}: {len(s)}" for d, s in enumerate(samples))} samples)')


def main():
    parser = argparse.ArgumentParser(description='Read the game clock from the score bug')
    commands = parser.add_subparsers(dest='command', required=True)
    templates_parser = commands.add_parser('templates', help='Create the digit templates')
    templates_parser.add_argument('video')
    templates_parser.add_argument('readings', nargs='+', help='FRAME=CLOCK, e.g. 200=9:57 or 28000=41.7')
    read_parser = commands.add_parser('read', help='Read the clock of every STEP-th frame')
    read_parser.add_argument('video')
    read_parser.add_argument('output')
    read_parser.add_argument('--step', type=int, default=5)
    args = parser.parse_args()

    if args.command == 'templates':
        make_templates(args.video, {int(f): text for f, text in (r.split('=') for r in args.readings)})
        return

    templates = load_templates()
    capture = cv2.VideoCapture(args.video)
    fps, length = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    clock = [[index, read_clock(frame, templates)] for index, frame in read_frames(args.video, range(0, length, args.step))]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'video': args.video, 'fps': fps, 'step': args.step, 'clock': clock}, f)
    read = sum(seconds is not None for _, seconds in clock)
    print(f'{read} of {len(clock)} frames read, saved to {args.output}')


if __name__ == '__main__':
    main()
