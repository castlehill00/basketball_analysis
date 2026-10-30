import bisect
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils import read_stub, save_stub, get_center_of_bbox


class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """
    def __init__(self, model_path, stretch_input=True):
        """
        Args:
            model_path (str): Path to the YOLO model weights.
            stretch_input (bool): Stretch frames to the model's square input size instead of
                letterboxing them (see detect_frames).
        """
        self.model = YOLO(model_path)
        self.stretch_input = stretch_input
        # Square size the model was trained at
        imgsz = self.model.overrides.get('imgsz', 640)
        self.input_size = imgsz if isinstance(imgsz, int) else max(imgsz)

    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.

        The ball model was trained on frames stretched to a square, so with stretch_input the
        frames are stretched the same way here. Letterboxing (scaling with padding, the YOLO
        default) shows the model a round ball where it learned a stretched one. The boxes are
        mapped back to frame coordinates in get_object_tracks.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size=20
        detections = []
        for i in range(0,len(frames),batch_size):
            batch = frames[i:i+batch_size]
            if self.stretch_input:
                batch = [cv2.resize(frame, (self.input_size, self.input_size)) for frame in batch]
            detections_batch = self.model.predict(batch,conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
        """
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        tracks=[]

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            if self.stretch_input:
                # Map boxes from the stretched square back to frame coordinates
                frame_height, frame_width = frames[frame_num].shape[:2]
                detection_supervision.xyxy = detection_supervision.xyxy * np.array(
                    [frame_width, frame_height, frame_width, frame_height]) / self.input_size

            tracks.append({})
            chosen_bbox =None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                
                if cls_id == cls_names_inv['Ball']:
                    if max_confidence<confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox":chosen_bbox}

        save_stub(stub_path,tracks)
        
        return tracks

    def remove_wrong_detections(self,ball_positions,rescue_detections_on_path=True):
        """
        Filter out incorrect ball detections by keeping the longest plausible ball trajectory.

        The ball moves at most maximum_allowed_distance pixels per frame, so two detections can
        belong to the same trajectory only if they are close enough for the frames between them.
        The longest chain of such detections over the whole video is found with dynamic
        programming. (Checking each detection against the last accepted one instead lets a
        single false detection lock out the real ball for many frames.)

        Detections left out of the chain are put back if they lie on the line between the kept
        detections around them: a fast pass or shot can beat the speed limit while still
        following a smooth path, which a head or arm detected as the ball does not.

        Args:
            ball_positions (list): List of detected ball positions across frames.
            rescue_detections_on_path (bool): Put back left-out detections that lie on the path.
                False keeps only the chain.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        maximum_allowed_distance = 25
        # After this many frames without a detection the ball could be anywhere in the frame
        maximum_link_gap = 60
        maximum_rescue_gap = 10
        maximum_rescue_distance = 20

        detected_frames = [i for i in range(len(ball_positions)) if len(ball_positions[i].get(1, {}).get('bbox', [])) > 0]
        centers = [np.array(get_center_of_bbox(ball_positions[i][1]['bbox'])) for i in detected_frames]

        # chain_length[j] is the length of the longest plausible chain ending at detection j
        chain_length = [1] * len(detected_frames)
        previous = [-1] * len(detected_frames)
        best_old_chain = (0, -1)
        window_start = 0
        for j in range(len(detected_frames)):
            # Chains that ended more than maximum_link_gap frames ago can always be continued
            while detected_frames[j] - detected_frames[window_start] > maximum_link_gap:
                if chain_length[window_start] > best_old_chain[0]:
                    best_old_chain = (chain_length[window_start], window_start)
                window_start += 1
            chain_length[j], previous[j] = best_old_chain[0] + 1, best_old_chain[1]

            for i in range(window_start, j):
                frame_gap = detected_frames[j] - detected_frames[i]
                distance = np.linalg.norm(centers[j] - centers[i])
                if distance <= maximum_allowed_distance * frame_gap and chain_length[i] + 1 > chain_length[j]:
                    chain_length[j], previous[j] = chain_length[i] + 1, i

        kept = set()
        j = int(np.argmax(chain_length)) if detected_frames else -1
        while j != -1:
            kept.add(j)
            j = previous[j]

        rescued = set()
        if rescue_detections_on_path:
            kept_sorted = sorted(kept)
            for k in range(len(detected_frames)):
                if k in kept:
                    continue
                position = bisect.bisect(kept_sorted, k)
                if position == 0 or position == len(kept_sorted):
                    continue
                i, j = kept_sorted[position - 1], kept_sorted[position]
                frame_gap = detected_frames[j] - detected_frames[i]
                if frame_gap > maximum_rescue_gap:
                    continue
                expected = centers[i] + (centers[j] - centers[i]) * (detected_frames[k] - detected_frames[i]) / frame_gap
                if np.linalg.norm(centers[k] - expected) <= maximum_rescue_distance:
                    rescued.add(k)

        for k, frame_num in enumerate(detected_frames):
            if k not in kept and k not in rescued:
                ball_positions[frame_num] = {}

        return ball_positions

    def interpolate_ball_positions(self,ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions