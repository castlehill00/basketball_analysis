import sys

import cv2
import numpy as np

sys.path.append('../')
from utils import read_stub, save_stub


class CameraCutDetector:
    """
    Splits a broadcast video into camera shots (the frames between two cuts).

    The court homography of one camera shot changes smoothly, but across a cut it jumps to an
    unrelated view, so it must only be smoothed within a shot. A cut is a frame whose colour
    histogram differs from the previous frame's much more than neighbouring frames differ from
    each other.
    """
    def __init__(self, min_distance=0.15, min_ratio=4.0, strong_distance=0.35, strong_ratio=2.5, neighbourhood=12):
        """
        Args:
            min_distance (float): A cut changes the histogram by at least this much (Bhattacharyya distance).
            min_ratio (float): ... and by this many times the usual change around it.
            strong_distance (float): A larger change counts as a cut when it is at least
                strong_ratio times the usual change; this catches cuts in busy scenes where the
                usual change is large too.
            strong_ratio (float): See strong_distance.
            neighbourhood (int): Number of frames on each side that set the usual change.
        """
        self.min_distance = min_distance
        self.min_ratio = min_ratio
        self.strong_distance = strong_distance
        self.strong_ratio = strong_ratio
        self.neighbourhood = neighbourhood

    def get_segments(self, frames, read_from_stub=False, stub_path=None):
        """
        Finds the camera shots of a video.

        The histogram changes are cached rather than the shots, so the thresholds can be changed
        without decoding the video again.

        Args:
            frames (sequence): Video frames.
            read_from_stub (bool): Whether to read cached histogram changes.
            stub_path (str): Path of the cache file.

        Returns:
            list: (first frame, last frame) of each camera shot.
        """
        distances = read_stub(read_from_stub, stub_path)
        if distances is None or len(distances) != len(frames):
            distances = self.get_histogram_distances(frames)
            save_stub(stub_path, distances)
        return self.find_segments(distances)

    def get_histogram_distances(self, frames):
        """
        Returns, for every frame, the Bhattacharyya distance between its HSV histogram and the
        previous frame's (0 for the first frame).
        """
        distances = np.zeros(len(frames), dtype=np.float32)
        previous = None
        for frame_num, frame in enumerate(frames):
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
            cv2.normalize(histogram, histogram)
            if previous is not None:
                distances[frame_num] = cv2.compareHist(previous, histogram, cv2.HISTCMP_BHATTACHARYYA)
            previous = histogram
        return distances

    def find_segments(self, distances):
        """
        Finds the cuts in the histogram changes.

        Args:
            distances (numpy.ndarray): Histogram change of every frame from get_histogram_distances.

        Returns:
            list: (first frame, last frame) of each camera shot.
        """
        distances = np.asarray(distances, dtype=np.float32)
        cuts = []
        for frame_num in range(1, len(distances) - 1):
            distance = distances[frame_num]
            if distance < distances[frame_num - 1] or distance < distances[frame_num + 1]:
                continue
            start, end = max(0, frame_num - self.neighbourhood), frame_num + self.neighbourhood + 1
            usual = np.median(np.r_[distances[start:frame_num], distances[frame_num + 1:end]])
            # The constant keeps a static scene (usual change near 0) from turning noise into cuts
            ratio = distance / (usual + 0.02)
            if (distance > self.min_distance and ratio > self.min_ratio) or \
                    (distance > self.strong_distance and ratio > self.strong_ratio):
                cuts.append(frame_num)

        bounds = [0] + cuts + [len(distances)]
        return [(first, next_first - 1) for first, next_first in zip(bounds, bounds[1:])]
