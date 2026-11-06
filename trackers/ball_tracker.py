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

        Yields:
            YOLO detection result for each frame, one batch at a time. The results are not
            collected in a list because each one keeps its (stretched) frame in memory.
        """
        batch_size=20
        for i in range(0,len(frames),batch_size):
            batch = frames[i:i+batch_size]
            if self.stretch_input:
                batch = [cv2.resize(frame, (self.input_size, self.input_size)) for frame in batch]
            yield from self.model.predict(batch,conf=0.5)

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
        return self.get_tracks_of_classes(frames, ['Ball'], read_from_stub, [stub_path])[0]

    def get_tracks_of_classes(self, frames, class_names, read_from_stub=False, stub_paths=None):
        """
        Get tracking results for several classes of the detector in one pass over the frames
        (the ball model also detects the hoop), keeping the most confident box of each class
        per frame.

        Args:
            frames (list): List of video frames to process.
            class_names (list): Names of the detector classes to track, e.g. ['Ball', 'Hoop'].
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_paths (list): Path to the cache file of each class.

        Returns:
            list: For each class, a list of dictionaries containing its tracking information for
                each frame.
        """
        stub_paths = stub_paths or [None] * len(class_names)
        cached = [read_stub(read_from_stub,stub_path) for stub_path in stub_paths]
        if all(tracks is not None and len(tracks) == len(frames) for tracks in cached):
            return cached

        # All frames have the same size
        frame_height, frame_width = frames[0].shape[:2]
        detections = self.detect_frames(frames)

        tracks_per_class = [[] for _ in class_names]

        for detection in detections:
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            if self.stretch_input:
                # Map boxes from the stretched square back to frame coordinates
                detection_supervision.xyxy = detection_supervision.xyxy * np.array(
                    [frame_width, frame_height, frame_width, frame_height]) / self.input_size

            for class_name, tracks in zip(class_names, tracks_per_class):
                tracks.append({})
                chosen_bbox =None
                max_confidence = 0

                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    confidence = frame_detection[2]

                    if cls_id == cls_names_inv[class_name]:
                        if max_confidence<confidence:
                            chosen_bbox = bbox
                            max_confidence = confidence

                if chosen_bbox is not None:
                    tracks[-1][1] = {"bbox":chosen_bbox}

        for stub_path, tracks in zip(stub_paths, tracks_per_class):
            save_stub(stub_path,tracks)

        return tracks_per_class

    def remove_wrong_detections(self,ball_positions,rescue_detections_on_path=True,camera_shots=None,
                                maximum_speed_in_ball_sizes=1.5,maximum_speed_in_pixels=None,ball_aspect_range=(0.6, 1.67)):
        """
        Filter out incorrect ball detections by keeping the longest plausible ball trajectory.

        Boxes that are not roughly square are removed first: the ball is round, while the
        detector also fires on wide overlay graphics such as the score bug.

        The ball moves at most maximum_speed_in_ball_sizes of its own diameter per frame (the
        mean box size of the two detections), so two detections can belong to the same
        trajectory only if they are close enough for the frames between them. A limit relative
        to the ball's size suits every zoom level; a fixed one in pixels (25 px per frame
        before) is too low for passes and shots seen close up. The longest chain of such
        detections is found with dynamic programming, separately within each camera shot when
        the shots are given: across a cut the ball can appear anywhere, so a chain must not
        continue there. (Checking each detection against the last accepted one instead lets a
        single false detection lock out the real ball for many frames.)

        Detections left out of the chain are put back if they lie on the line between the kept
        detections around them: a fast pass or shot can beat the speed limit while still
        following a smooth path, which a head or arm detected as the ball does not.

        Args:
            ball_positions (list): List of detected ball positions across frames.
            rescue_detections_on_path (bool): Put back left-out detections that lie on the path.
                False keeps only the chain.
            camera_shots (list or None): (first frame, last frame) of each camera shot, from
                CameraCutDetector.get_segments; None treats the video as one shot.
            maximum_speed_in_ball_sizes (float): Speed limit in ball diameters per frame.
            maximum_speed_in_pixels (float or None): A fixed speed limit in pixels per frame
                instead (25 reproduces the earlier filter).
            ball_aspect_range (tuple or None): Allowed width/height ratios of a ball box; None
                skips the shape check.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        if ball_aspect_range is not None:
            for frame_num, positions in enumerate(ball_positions):
                bbox = positions.get(1, {}).get('bbox', [])
                if len(bbox) == 0:
                    continue
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if height <= 0 or not ball_aspect_range[0] <= width / height <= ball_aspect_range[1]:
                    ball_positions[frame_num] = {}

        for first_frame, last_frame in camera_shots or [(0, len(ball_positions) - 1)]:
            # Called on the class too (without an instance) by the evaluation scripts
            BallTracker.keep_longest_chain(ball_positions, first_frame, last_frame, rescue_detections_on_path,
                                           maximum_speed_in_ball_sizes, maximum_speed_in_pixels)
        return ball_positions

    @staticmethod
    def keep_longest_chain(ball_positions, first_frame, last_frame, rescue_detections_on_path, maximum_speed_in_ball_sizes,
                           maximum_speed_in_pixels=None):
        """Keeps the longest plausible chain of detections between two frames (see remove_wrong_detections)."""
        # After this many frames without a detection the ball could be anywhere in the frame
        maximum_link_gap = 60
        maximum_rescue_gap = 10
        maximum_rescue_distance = 20

        detected_frames = [i for i in range(first_frame, last_frame + 1) if len(ball_positions[i].get(1, {}).get('bbox', [])) > 0]
        boxes = [ball_positions[i][1]['bbox'] for i in detected_frames]
        centers = [np.array(get_center_of_bbox(bbox)) for bbox in boxes]
        sizes = [max(bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in boxes]

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
                speed = maximum_speed_in_pixels or maximum_speed_in_ball_sizes * (sizes[i] + sizes[j]) / 2
                maximum_distance = speed * frame_gap
                if distance <= maximum_distance and chain_length[i] + 1 > chain_length[j]:
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

    def interpolate_ball_positions(self,ball_positions,camera_shots=None,max_gap=None):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Without camera shots and a maximum gap this fills every gap and the frames before the
        first detection, as the original code did; the ball is then shown in every frame,
        including close-ups and replays without a ball. Given camera shots, gaps are only filled
        within a shot, and never before the first or after the last detection of a shot.

        Args:
            ball_positions (list): List of ball positions with potential gaps.
            camera_shots (list or None): (first frame, last frame) of each camera shot.
            max_gap (int or None): Longest gap (in frames) that is filled; None fills any gap.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        if camera_shots is None and max_gap is None:
            ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
            df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

            # Interpolate missing values
            df_ball_positions = df_ball_positions.interpolate()
            df_ball_positions = df_ball_positions.bfill()

            ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
            return ball_positions

        interpolated = [dict(positions) for positions in ball_positions]
        for first_frame, last_frame in camera_shots or [(0, len(ball_positions) - 1)]:
            known = [i for i in range(first_frame, last_frame + 1) if len(ball_positions[i].get(1, {}).get('bbox', [])) > 0]
            for start, end in zip(known, known[1:]):
                gap = end - start - 1
                if gap == 0 or (max_gap is not None and gap > max_gap):
                    continue
                start_box = np.array(ball_positions[start][1]['bbox'], dtype=float)
                end_box = np.array(ball_positions[end][1]['bbox'], dtype=float)
                for frame_num in range(start + 1, end):
                    share = (frame_num - start) / (end - start)
                    interpolated[frame_num] = {1: {"bbox": (start_box + (end_box - start_box) * share).tolist()}}
        return interpolated