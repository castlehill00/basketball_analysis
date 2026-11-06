"""
A module for reading and writing video files.

This module provides utility functions to load video frames into memory and save
processed frames back to video files, with support for common video formats.
"""

import cv2
import os


class VideoFrames:
    """
    The frames of a video file, decoded when they are accessed instead of all at once.

    Supports len(), indexing, slicing and iteration like the list read_video returns, so it can
    be used wherever a list of frames is expected, but only the frames in use are in memory
    (as a list, one minute of 720p video takes about 4 GB). Frames are read fastest in order;
    going back to an earlier frame decodes the video again from the start, because seeking by
    frame number is not always exact.
    """
    def __init__(self, video_path):
        """
        Args:
            video_path (str): Path to the input video file.
        """
        self.video_path = video_path
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise IOError(f'Cannot open video {video_path}')
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        self.capture = None
        # Index of the frame the capture returns next
        self.next_index = 0
        # The frame read last, which is often asked for again (for example once per player)
        self.last_index, self.last_frame = None, None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.length))]
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(f'Frame {index} is out of range for a video of {self.length} frames')
        if index == self.last_index:
            return self.last_frame

        if self.capture is None or index < self.next_index:
            self.capture = cv2.VideoCapture(self.video_path)
            self.next_index = 0
        while self.next_index < index:
            self.capture.grab()
            self.next_index += 1
        ok, frame = self.capture.read()
        self.next_index += 1
        if not ok:
            # The header can claim a few frames more than the file has; repeat the last frame
            # so that every per-frame list still has one entry per frame
            if self.last_frame is None:
                raise IOError(f'Cannot read frame {index} of {self.video_path}')
            print(f'Warning: frame {index} of {self.video_path} could not be read, repeating the previous frame')
            frame = self.last_frame
        self.last_index, self.last_frame = index, frame
        return frame

    def __iter__(self):
        for index in range(self.length):
            yield self[index]


def read_video(video_path):
    """
    Read all frames from a video file into memory.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: List of video frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path,fps=24):
    """
    Save a sequence of frames as a video file.

    Creates necessary directories if they don't exist and writes frames using XVID codec.

    Args:
        ouput_video_frames (iterable): Frames to save. A generator works too, so that the frames
            never have to be in memory together.
        output_video_path (str): Path where the video should be saved.
        fps (float): Frame rate of the saved video (use the input video's).
    """
    # If folder doesn't exist, create it
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    for frame in ouput_video_frames:
        if out is None:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            # VideoWriter does not raise when it cannot write the file (for example while it is
            # open in a video player on Windows); it would silently write nothing
            if not out.isOpened():
                raise IOError(f'Cannot write the video {output_video_path} (is it open in another program?)')
        out.write(frame)
    if out is not None:
        out.release()