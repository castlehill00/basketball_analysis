import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from ultralytics.engine.results import Keypoints
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import get_foot_position,measure_distance

class TacticalViewConverter:
    def __init__(self, court_image_path,
                 keypoint_conf_threshold=0.5,
                 ransac_threshold_in_meters=1.0,
                 fix_mirrored_labels=True,
                 smoothing_window=5):
        """
        Args:
            court_image_path (str): Path to the court image used for the tactical view.
            keypoint_conf_threshold (float or None): Court keypoints below this confidence are
                treated as undetected. None keeps every keypoint.
            ransac_threshold_in_meters (float or None): How far (on the court) a keypoint may be
                from where the homography puts it and still count as a RANSAC inlier. None fits a
                plain least-squares homography to all keypoints, as the original code did.
            fix_mirrored_labels (bool): Relabel keypoints that were given to the wrong half of
                the court (see estimate_frame_homography).
            smoothing_window (int): Number of neighbouring frames the homography is smoothed
                over. 1 disables smoothing.
        """
        self.court_image_path = court_image_path
        self.width = 300
        self.height= 161

        self.actual_width_in_meters=28
        self.actual_height_in_meters=15

        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.ransac_threshold_in_meters = ransac_threshold_in_meters
        self.fix_mirrored_labels = fix_mirrored_labels
        self.smoothing_window = smoothing_window
        # Keypoints closer than this to a line count as lying on it (see is_degenerate)
        self.collinear_tolerance_in_meters = 0.5

        self.key_points = [
            # left edge
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height)),
            (0,int((5.18/self.actual_height_in_meters)*self.height)),
            (0,int((10/self.actual_height_in_meters)*self.height)),
            (0,int((14.1/self.actual_height_in_meters)*self.height)),
            (0,int(self.height)),

            # Middle line
            (int(self.width/2),self.height),
            (int(self.width/2),0),
            
            # Left Free throw line
            (int((5.79/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),

            # right edge
            (self.width,int(self.height)),
            (self.width,int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width,int((10/self.actual_height_in_meters)*self.height)),
            (self.width,int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width,int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width,0),

            # Right Free throw line
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),
        ]

        # Court lines as pairs of key_points indices, used to draw the court back onto the video
        self.court_lines = [
            (0, 5), (5, 10), (10, 15), (15, 0),  # outer boundary
            (7, 6),                              # half-court line
            (2, 8), (8, 9), (9, 3),              # left lane
            (13, 16), (16, 17), (17, 12),        # right lane
        ]

        # Index of the same keypoint on the other half of the court (mirrored left to right)
        self.mirrored_key_point_indices = [
            int(np.argmin([measure_distance((self.width - x, y), key_point) for key_point in self.key_points]))
            for x, y in self.key_points
        ]

    def filter_low_confidence_keypoints(self, keypoints_list):
        """
        Marks court keypoints the detector is unsure about as undetected.

        The pose model outputs all 18 keypoints in every frame, including off-screen or hidden
        ones, which come with a low confidence and a guessed position. ultralytics up to 8.3.x
        set keypoints below 0.5 confidence to (0, 0) itself; newer versions keep them, so the
        filtering is done here to give the same result whatever the library version.

        Args:
            keypoints_list (list): Court keypoints (ultralytics Keypoints) for each frame.

        Returns:
            list: Keypoints per frame where keypoints below keypoint_conf_threshold are (0, 0).
        """
        if self.keypoint_conf_threshold is None:
            return keypoints_list

        filtered_keypoints = []
        for frame_keypoints in keypoints_list:
            data = deepcopy(frame_keypoints.data)
            if data.shape[-1] == 3:
                data[..., :2][data[..., 2] < self.keypoint_conf_threshold] = 0
            filtered_keypoints.append(Keypoints(data, frame_keypoints.orig_shape))
        return filtered_keypoints

    def validate_keypoints(self, keypoints_list):
        """
        Validates detected keypoints by comparing their proportional distances
        to the tactical view keypoints.
        
        Args:
            keypoints_list (List[List[Tuple[float, float]]]): A list containing keypoints for each frame.
                Each outer list represents a frame.
                Each inner list contains keypoints as (x, y) tuples.
                A keypoint of (0, 0) indicates that the keypoint is not detected for that frame.
        
        Returns:
            List[bool]: A list indicating whether each frame's keypoints are valid.
        """

        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            # Skip frames where no court was detected (e.g. close-ups)
            if len(frame_keypoints.xy) == 0:
                continue
            frame_keypoints = frame_keypoints.xy.tolist()[0]
            
            # Get indices of detected keypoints (not (0, 0))
            detected_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] >0 and kp[1]>0]
            
            # Need at least 3 detected keypoints to validate proportions
            if len(detected_indices) < 3:
                continue
            
            invalid_keypoints = []
            # Validate each detected keypoint
            for i in detected_indices:
                # Skip if this is (0, 0)
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                # Choose two other random detected keypoints
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                # Take first two other indices for simplicity
                j, k = other_indices[0], other_indices[1]

                # Calculate distances between detected keypoints
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                
                # Calculate distances between corresponding tactical keypoints
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                # Calculate and compare proportions with 50% error margin
                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = (prop_detected - prop_tactical) / prop_tactical
                    error = abs(error)

                    if error >0.8:  # 80% error margin                        
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)
            
        return keypoints_list

    def get_homographies(self, keypoints_list):
        """
        Estimates the homography from video frame to tactical view for every frame.

        Each frame is solved on its own first, then the results are smoothed over neighbouring
        frames, which also fills short gaps where a frame alone lacked reliable keypoints.

        Args:
            keypoints_list (list): Court keypoints (ultralytics Keypoints) for each frame.

        Returns:
            list: A 3x3 homography matrix per frame, or None where no reliable homography exists.
        """
        homographies = [self.estimate_frame_homography(frame_keypoints) for frame_keypoints in keypoints_list]

        if self.smoothing_window > 1 and len(keypoints_list) > 0:
            frame_height, frame_width = keypoints_list[0].orig_shape
            homographies = self.smooth_homographies(homographies, frame_width, frame_height)

        return homographies

    def estimate_frame_homography(self, frame_keypoints):
        """
        Estimates the homography of a single frame from its detected court keypoints.

        The two halves of the court look alike, and the detector sometimes gives the half it
        sees the labels of the other half. That yields a mirror-image homography; when
        fix_mirrored_labels is set, the keypoints are relabelled as the other half and refit.

        Args:
            frame_keypoints (Keypoints): Court keypoints detected in the frame.

        Returns:
            numpy.ndarray or None: The 3x3 homography matrix, or None if it cannot be estimated reliably.
        """
        if frame_keypoints is None or len(frame_keypoints.xy) == 0:
            return None

        frame_keypoints = frame_keypoints.xy.tolist()[0]

        # Filter out undetected keypoints (those with coordinates (0,0))
        valid_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0]

        # Need at least 4 points for a homography
        if len(valid_indices) < 4:
            return None

        source_points = np.array([frame_keypoints[i] for i in valid_indices], dtype=np.float32)
        homography = self.fit_homography(source_points, valid_indices)

        if self.fix_mirrored_labels and homography is not None and self.is_mirrored(homography, source_points):
            mirrored_indices = [self.mirrored_key_point_indices[i] for i in valid_indices]
            homography = self.fit_homography(source_points, mirrored_indices)
            if homography is not None and self.is_mirrored(homography, source_points):
                return None

        return homography

    def fit_homography(self, source_points, key_point_indices):
        """
        Fits a homography from detected keypoints to their positions in the tactical view.

        Without a RANSAC threshold this is a plain least-squares fit, as in the original code.
        With one, keypoints that disagree with the others (a misplaced detection) are left out,
        and the fit is rejected when the remaining keypoints cannot pin down a homography.

        Args:
            source_points (numpy.ndarray): Keypoint positions in the frame.
            key_point_indices (list): Index into self.key_points for each source point.

        Returns:
            numpy.ndarray or None: The 3x3 homography matrix, or None if the fit is unreliable.
        """
        target_points = np.array([self.key_points[i] for i in key_point_indices], dtype=np.float32)

        ransac_threshold = None
        if self.ransac_threshold_in_meters is not None:
            ransac_threshold = self.meters_to_pixels(self.ransac_threshold_in_meters)

        try:
            homography = Homography(source_points, target_points, ransac_threshold)
        except (ValueError, cv2.error):
            return None

        if ransac_threshold is not None:
            inlier_points = target_points[homography.inliers]
            if len(inlier_points) < 4 or self.is_degenerate(inlier_points):
                return None

        return homography.m

    def is_mirrored(self, homography, image_points):
        """
        Checks whether a homography maps the frame onto a mirror image of the court.

        Seen from a camera above it, the court keeps its orientation in the tactical view
        (clockwise stays clockwise), so the homography's Jacobian determinant is positive.
        It is checked at the centre of the detected keypoints, which lie on the court.

        Args:
            homography (numpy.ndarray): Frame to tactical view homography.
            image_points (numpy.ndarray): Detected keypoint positions in the frame.

        Returns:
            bool: True if the homography flips the court.
        """
        center_x, center_y = image_points.mean(axis=0)
        w = homography[2] @ np.array([center_x, center_y, 1.0])
        # The Jacobian determinant of a homography at a point is det(H) / w^3
        return np.linalg.det(homography) * w < 0

    def is_degenerate(self, tactical_points):
        """
        Checks whether keypoints lie too close to a single court line to fix a homography.

        Many court keypoints share a line, e.g. the six points on each baseline. A homography
        needs at least two points away from any line through the others; otherwise everything
        off that line is extrapolated and can end up metres away.

        Args:
            tactical_points (numpy.ndarray): Keypoint positions in tactical view coordinates.

        Returns:
            bool: True if the points cannot determine a homography reliably.
        """
        tolerance = self.meters_to_pixels(self.collinear_tolerance_in_meters)

        for i in range(len(tactical_points)):
            for j in range(i + 1, len(tactical_points)):
                direction = tactical_points[j] - tactical_points[i]
                length = np.linalg.norm(direction)
                if length == 0:
                    continue

                # Distance of every point to the line through points i and j
                offsets = tactical_points - tactical_points[i]
                distances = np.abs(direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]) / length
                if np.sum(distances > tolerance) < 2:
                    return True

        return False

    def smooth_homographies(self, homographies, frame_width, frame_height):
        """
        Smooths homographies over time and fills short gaps.

        Averaging matrix entries has no geometric meaning, so a fixed grid of image points is
        projected with every homography in the window, each grid point takes the median of its
        projections, and the homography is refit from those medians. The median also outvotes
        a single bad frame.

        Args:
            homographies (list): A 3x3 homography matrix or None per frame.
            frame_width (int): Width of the video frames in pixels.
            frame_height (int): Height of the video frames in pixels.

        Returns:
            list: The smoothed homography matrix or None per frame.
        """
        # The court usually covers the lower part of a broadcast frame
        grid = np.array([[x * frame_width, y * frame_height]
                         for x in (0.1, 0.5, 0.9)
                         for y in (0.4, 0.7, 1.0)], dtype=np.float32)

        projected_grids = []
        for homography in homographies:
            if homography is None:
                projected_grids.append(None)
                continue
            projected_grids.append(cv2.perspectiveTransform(grid.reshape(-1, 1, 2), homography).reshape(-1, 2))

        half_window = self.smoothing_window // 2
        smoothed_homographies = []
        for frame_idx in range(len(homographies)):
            window = projected_grids[max(0, frame_idx - half_window):frame_idx + half_window + 1]
            window = [projected_grid for projected_grid in window if projected_grid is not None]
            if len(window) == 0:
                smoothed_homographies.append(None)
                continue

            median_grid = np.median(np.stack(window), axis=0).astype(np.float32)
            smoothed_homography, _ = cv2.findHomography(grid, median_grid)
            smoothed_homographies.append(smoothed_homography)

        return smoothed_homographies

    def transform_players_to_tactical_view(self, homographies, player_tracks):
        """
        Transform player positions from video frame coordinates to tactical view coordinates.

        Args:
            homographies (list): Homography matrix (or None) for each frame, from get_homographies.
            player_tracks (list): List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.

        Returns:
            list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                in the tactical view coordinate system. The list index corresponds to the frame number.
        """
        tactical_player_positions = []

        for homography, frame_tracks in zip(homographies, player_tracks):
            # Initialize empty dictionary for this frame
            tactical_positions = {}

            if homography is None:
                tactical_player_positions.append(tactical_positions)
                continue

            # Transform each player's position
            for player_id, player_data in frame_tracks.items():
                bbox = player_data["bbox"]
                # Use bottom center of bounding box as player position
                player_position = np.array([[get_foot_position(bbox)]], dtype=np.float32)
                # Transform to tactical view coordinates
                tactical_position = cv2.perspectiveTransform(player_position, homography)[0][0]

                # If tactical position is not in the tactical view, skip
                if tactical_position[0] < 0 or tactical_position[0] > self.width or tactical_position[1] < 0 or tactical_position[1] > self.height:
                    continue

                tactical_positions[player_id] = tactical_position.tolist()

            tactical_player_positions.append(tactical_positions)

        return tactical_player_positions

    def meters_to_pixels(self, meters):
        """
        Converts a distance on the court from meters to tactical view pixels.
        """
        return meters * self.width / self.actual_width_in_meters
