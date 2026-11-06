import os
import sys
import pathlib
import warnings
import numpy as np
import cv2
from copy import deepcopy
from ultralytics.engine.results import Keypoints
from .homography import Homography
from .court_dimensions import COURT_DIMENSIONS

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import get_foot_position,measure_distance

class TacticalViewConverter:
    def __init__(self, court_image_path,
                 league='fiba',
                 keypoint_conf_threshold=0.5,
                 ransac_threshold_in_meters=1.0,
                 min_inliers=5,
                 max_inlier_error_in_meters=0.2,
                 fix_mirrored_labels=True,
                 smoothing_window=5,
                 smoothing_half_window=6):
        """
        Args:
            court_image_path (str): Path to the court image used for the tactical view.
            league (str): 'fiba' (also the CBA) or 'nba'; sets the court dimensions.
            keypoint_conf_threshold (float or None): Court keypoints below this confidence are
                treated as undetected. None keeps every keypoint.
            ransac_threshold_in_meters (float or None): How far (on the court) a keypoint may be
                from where the homography puts it and still count as a RANSAC inlier. None fits a
                plain least-squares homography to all keypoints, as the original code did.
            min_inliers (int): A RANSAC fit must agree with at least this many keypoints. Any
                four keypoints fit some homography exactly, so a fit to four is not checked by
                anything; on the 2023 World Cup final most such fits were wrong.
            max_inlier_error_in_meters (float or None): Largest root-mean-square distance (on the
                court) of the inlier keypoints from where the homography puts them. None
                disables this check.
            fix_mirrored_labels (bool): Relabel keypoints that were given to the wrong half of
                the court (see estimate_frame_homography).
            smoothing_window (int): Without camera shots (see get_homographies), the number of
                neighbouring frames the homography is smoothed over. 1 disables smoothing.
            smoothing_half_window (int): With camera shots, the number of frames on each side
                the homography is smoothed over (see smooth_camera_shot). Longer windows smooth
                more but lag behind fast pans: on the sample videos 9 or more frames already
                lowered the share of keypoints the homography agrees with.
        """
        self.court_image_path = court_image_path
        self.court = COURT_DIMENSIONS[league]

        self.actual_width_in_meters=self.court.length
        self.actual_height_in_meters=self.court.width

        self.width = 300
        self.height= round(self.width * self.actual_height_in_meters / self.actual_width_in_meters)

        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.ransac_threshold_in_meters = ransac_threshold_in_meters
        self.min_inliers = min_inliers
        self.max_inlier_error_in_meters = max_inlier_error_in_meters
        self.fix_mirrored_labels = fix_mirrored_labels
        self.smoothing_window = smoothing_window
        self.smoothing_half_window = smoothing_half_window
        # Keypoints closer than this to a line count as lying on it (see is_degenerate)
        self.collinear_tolerance_in_meters = 0.5
        # A camera shot gets homographies only if at least this share of its frames have one of
        # their own and at least this share of those agree with the smoothed homography
        self.min_shot_coverage = 0.5
        self.min_shot_agreement = 0.7
        # A frame's homography disagrees with the smoothed one when the court points it projects
        # are this far (median, in image pixels) from where the smoothed one projects them
        self.outlier_pixels = 20
        # A frame gets a smoothed homography only where the fit rests on enough agreeing frames:
        # their weights in the fit (time kernel times agreement) must add up to this much. A full
        # window of agreeing frames adds up to about 8; one frame on its own to at most 1.
        self.min_fit_support = 3.0
        # Gaps of up to this many frames between frames with enough support are bridged
        self.max_gap_frames = 12

        # Court keypoints in tactical view pixels (whole pixels, as the tactical view drawer expects)
        self.key_points = [tuple(int(round(v)) for v in self.meters_to_tactical(point))
                           for point in self.court.key_points()]

        # Court markings in tactical view pixels, used to draw the court back onto the video
        self.court_polylines = [self.meters_to_tactical(line) for line in self.court.lines()]

        # Index of the same keypoint on the other half of the court (mirrored left to right)
        self.mirrored_key_point_indices = [
            int(np.argmin([measure_distance((self.width - x, y), key_point) for key_point in self.key_points]))
            for x, y in self.key_points
        ]

        # Court points whose image positions represent a homography when smoothing (tactical view pixels)
        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width, 15), np.linspace(0, self.height, 7))
        self.anchor_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

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

    def get_frame_homographies(self, keypoints_list):
        """
        Estimates the homography of every frame on its own (see estimate_frame_homography).

        Args:
            keypoints_list (list): Court keypoints (ultralytics Keypoints) for each frame.

        Returns:
            list: A 3x3 homography matrix per frame, or None.
        """
        return [self.estimate_frame_homography(frame_keypoints) for frame_keypoints in keypoints_list]

    def get_homographies(self, keypoints_list, camera_shots=None, frame_homographies=None, check_shots=True):
        """
        Estimates the homography from video frame to tactical view for every frame.

        Each frame is solved on its own first. Given the camera shots of the video, each shot is
        then checked and smoothed on its own (see smooth_camera_shot). Without them, the results
        are smoothed over a few neighbouring frames, which also fills short gaps where a frame
        alone lacked reliable keypoints.

        Args:
            keypoints_list (list): Court keypoints (ultralytics Keypoints) for each frame.
            camera_shots (list or None): (first frame, last frame) of each camera shot, from
                CameraCutDetector.get_segments.
            frame_homographies (list or None): Per-frame homographies to smooth instead of the
                ones estimated from the keypoints (e.g. adjusted by CourtLineRefiner).
            check_shots (bool): Whether a camera shot must pass the checks of smooth_camera_shot to
                get homographies; off for homographies that were already checked.

        Returns:
            list: A 3x3 homography matrix per frame, or None where no reliable homography exists.
        """
        homographies = list(frame_homographies) if frame_homographies is not None \
            else self.get_frame_homographies(keypoints_list)
        if len(keypoints_list) == 0:
            return homographies
        frame_height, frame_width = keypoints_list[0].orig_shape

        if camera_shots is not None:
            smoothed_homographies = [None] * len(homographies)
            for first_frame, last_frame in camera_shots:
                smoothed = self.smooth_camera_shot(homographies, first_frame, last_frame, frame_width, frame_height,
                                                   check_shots)
                for frame_num, homography in smoothed.items():
                    smoothed_homographies[frame_num] = homography
            return smoothed_homographies

        if self.smoothing_window > 1:
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
            if len(inlier_points) < max(4, self.min_inliers) or self.is_degenerate(inlier_points):
                return None
            if self.max_inlier_error_in_meters is not None:
                projected = cv2.perspectiveTransform(source_points[homography.inliers].reshape(-1, 1, 2),
                                                     homography.m).reshape(-1, 2)
                error = np.sqrt(np.mean(np.sum((projected - inlier_points) ** 2, axis=1)))
                if error > self.meters_to_pixels(self.max_inlier_error_in_meters):
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

    def smooth_camera_shot(self, homographies, first_frame, last_frame, frame_width, frame_height, check=True):
        """
        Checks and smooths the homographies of one camera shot.

        The keypoint model was trained on the main broadcast camera. In other views (close-ups,
        the cameras behind the baskets or low at the side) it still finds keypoints now and then,
        and the homographies fitted to them jump from frame to frame. A shot therefore gets
        homographies only if most of its frames have one and most of those agree with each other.

        Each homography is represented by where it puts a grid of court points in the image,
        where keypoint errors arise and are of similar size everywhere (in court coordinates a
        point near the far sideline can be off by metres). These positions are smoothed over time
        by a robust local linear fit (LOWESS): the frames near in time are fitted with a straight
        line, which follows a panning camera without lag, and frames far from the fit get less
        weight in the next round, so single wrong frames are ignored. The rounds start from the
        median position over the window, not from a first straight-line fit to all frames: a
        single frame hundreds of pixels off (keypoints given the labels of both baskets at once)
        drags such a fit along, its neighbours then look wrong too, and the fit ends up resting on
        one or two frames at the edge of the window.

        Frames of the shot get the homography refitted to the smoothed positions where the fit
        rests on enough agreeing frames (min_fit_support), which also fills frames that had none.
        Where it does not, the fit follows whichever few frames are in the window and jumps when
        one of them enters or leaves it; short gaps between supported frames are bridged instead
        (see bridge_gap), longer ones get no homography.

        Args:
            homographies (list): Per-frame homography (or None) of the whole video.
            first_frame (int): First frame of the shot.
            last_frame (int): Last frame of the shot.
            frame_width (int): Width of the video frames in pixels.
            frame_height (int): Height of the video frames in pixels.
            check (bool): Whether to apply the coverage and agreement checks. Without them only the
                smoothing is done, and frames are compared on the court points inside the frame
                (for homographies adjusted to the painted lines, which may differ off screen).

        Returns:
            dict: Frame number -> smoothed homography, for the frames of the shot that get one.
        """
        frame_nums = np.array([t for t in range(first_frame, last_frame + 1) if homographies[t] is not None])
        if len(frame_nums) < 3 or (check and len(frame_nums) < self.min_shot_coverage * (last_frame - first_frame + 1)):
            return {}

        positions = np.stack([self.project_anchor_points(homographies[t], frame_width, frame_height)
                              for t in frame_nums])

        def agreement(reference_positions):
            """Tukey's biweight of each frame's distance from the reference: full weight near it, none beyond outlier_pixels."""
            residuals = np.array([self.median_distance(frame_positions, reference, frame_width, frame_height,
                                                       margin=0.5 if check else 0.0)
                                  for frame_positions, reference in zip(positions, reference_positions)])
            return np.clip(1 - (residuals / self.outlier_pixels) ** 2, 0, 1) ** 2

        weights = agreement(self.running_median(frame_nums, positions))
        for _ in range(3):
            weights = agreement(self.local_linear_fit(frame_nums, positions, weights, frame_nums))
        if check and np.mean(weights > 0) < self.min_shot_agreement:
            return {}
        if not np.any(weights > 0):
            return {}

        shot_frames = np.arange(first_frame, last_frame + 1)
        supported = shot_frames[self.fit_support(frame_nums, weights, shot_frames) >= self.min_fit_support]
        if len(supported) == 0:
            return {}
        fitted = dict(zip(supported, self.local_linear_fit(frame_nums, positions, weights, supported)))
        bridged = {}
        for before, after in zip(supported[:-1], supported[1:]):
            if 1 < after - before <= self.max_gap_frames + 1:
                bridged.update(self.bridge_gap(fitted, before, after))
        fitted.update(bridged)

        smoothed = {}
        for frame_num in sorted(fitted):
            homography = self.fit_to_anchor_points(fitted[frame_num], frame_width, frame_height)
            if homography is not None:
                smoothed[int(frame_num)] = homography
        return smoothed

    def project_anchor_points(self, homography, frame_width, frame_height):
        """
        Returns the image positions of the anchor points (nan for points behind the camera, and
        for all of them if the homography cannot be inverted).

        A homography is only defined up to scale, so which sign of the projective coordinate
        means "in front of the camera" is taken from the bottom centre of the frame, which shows
        the floor in every view a homography can come from.
        """
        try:
            court_to_frame = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return np.full((len(self.anchor_points), 2), np.nan)
        reference = homography @ np.array([frame_width / 2, frame_height, 1.0])
        if reference[2] < 0:
            court_to_frame = -court_to_frame
        projected = np.c_[self.anchor_points, np.ones(len(self.anchor_points))] @ court_to_frame.T
        with np.errstate(divide='ignore', invalid='ignore'):
            positions = projected[:, :2] / projected[:, 2:3]
        positions[projected[:, 2] <= 0] = np.nan
        return positions

    def running_median(self, frame_nums, positions):
        """
        Median of each anchor's image position over the frames within smoothing_half_window of
        each frame (nan where none of them has it).

        Args:
            frame_nums (numpy.ndarray): Frames that have a homography, shape (n,).
            positions (numpy.ndarray): Their anchor image positions, shape (n, anchors, 2), nan where unknown.

        Returns:
            numpy.ndarray: Median positions, shape (n, anchors, 2).
        """
        medians = np.full(positions.shape, np.nan)
        with warnings.catch_warnings():
            # An anchor behind the camera in every frame of a window gives an all-nan slice
            warnings.simplefilter('ignore', RuntimeWarning)
            for index, frame_num in enumerate(frame_nums):
                lo = np.searchsorted(frame_nums, frame_num - self.smoothing_half_window)
                hi = np.searchsorted(frame_nums, frame_num + self.smoothing_half_window, side='right')
                medians[index] = np.nanmedian(positions[lo:hi], axis=0)
        return medians

    @staticmethod
    def bridge_gap(fitted, before, after):
        """
        Anchor positions for the frames between two fitted frames, along a cubic that leaves the
        first and joins the second with the velocity the fit has there (Hermite interpolation;
        the velocity comes from the neighbouring fitted frame, or is the straight line across the
        gap where there is none). A straight line would turn sharply at both ends during a pan.

        Args:
            fitted (dict): Frame number -> fitted anchor positions, shape (anchors, 2).
            before (int): Last fitted frame before the gap.
            after (int): First fitted frame after the gap.

        Returns:
            dict: Frame number -> anchor positions, for the frames of the gap.
        """
        start, end, span = fitted[before], fitted[after], after - before
        start_velocity = start - fitted[before - 1] if before - 1 in fitted else (end - start) / span
        end_velocity = fitted[after + 1] - end if after + 1 in fitted else (end - start) / span
        positions = {}
        for frame_num in range(before + 1, after):
            s = (frame_num - before) / span
            positions[frame_num] = ((2 * s ** 3 - 3 * s ** 2 + 1) * start + (s ** 3 - 2 * s ** 2 + s) * span * start_velocity
                                    + (-2 * s ** 3 + 3 * s ** 2) * end + (s ** 3 - s ** 2) * span * end_velocity)
        return positions

    def time_kernel(self, frame_offsets):
        """Tricube weight of frames by their distance in time from the target; zero beyond smoothing_half_window."""
        return np.clip(1 - (np.abs(frame_offsets) / (self.smoothing_half_window + 1)) ** 3, 0, None) ** 3

    def fit_support(self, frame_nums, weights, target_frames):
        """
        How much the local linear fit at each target frame rests on: the time kernel times the
        weight of every frame near it, summed.

        Args:
            frame_nums (numpy.ndarray): Frames that have a homography, shape (n,).
            weights (numpy.ndarray): Weight of each of these frames, shape (n,).
            target_frames (numpy.ndarray): Frames to evaluate the support at, shape (m,).

        Returns:
            numpy.ndarray: Support of each target frame, shape (m,).
        """
        support = np.zeros(len(target_frames))
        for start in range(0, len(target_frames), 200):
            targets = target_frames[start:start + 200]
            lo = np.searchsorted(frame_nums, targets[0] - self.smoothing_half_window)
            hi = np.searchsorted(frame_nums, targets[-1] + self.smoothing_half_window, side='right')
            support[start:start + len(targets)] = self.time_kernel(frame_nums[lo:hi][None, :] - targets[:, None]) @ weights[lo:hi]
        return support

    @staticmethod
    def near_frame(positions, frame_width, frame_height, margin=0.5):
        """Which image positions lie inside the frame enlarged on every side by margin (a share of its size)."""
        x, y = positions[:, 0], positions[:, 1]
        with np.errstate(invalid='ignore'):
            return (np.isfinite(positions).all(axis=1) & (x > -margin * frame_width) & (x < (1 + margin) * frame_width)
                    & (y > -margin * frame_height) & (y < (1 + margin) * frame_height))

    def median_distance(self, positions, fitted_positions, frame_width, frame_height, margin=0.5):
        """Median image distance of a frame's anchor positions from the fitted ones, over anchors near the frame."""
        use = self.near_frame(fitted_positions, frame_width, frame_height, margin) & np.isfinite(positions).all(axis=1)
        if use.sum() < 4:
            return np.inf
        return float(np.median(np.linalg.norm(positions[use] - fitted_positions[use], axis=1)))

    def local_linear_fit(self, frame_nums, positions, weights, target_frames):
        """
        Fits each anchor's image position over time with a weighted straight line around each
        target frame and returns the line's value at that frame.

        Frames within smoothing_half_window of the target are weighted by a tricube of their
        distance in time times their own weight. With fewer than three frames, or all at the
        same time, the weighted mean is used instead.

        Args:
            frame_nums (numpy.ndarray): Frames that have a homography, shape (n,).
            positions (numpy.ndarray): Their anchor image positions, shape (n, anchors, 2), nan where unknown.
            weights (numpy.ndarray): Weight of each of these frames, shape (n,).
            target_frames (numpy.ndarray): Frames to evaluate the fit at, shape (m,).

        Returns:
            numpy.ndarray: Fitted positions, shape (m, anchors, 2); nan where no frame is near.
        """
        fitted = np.full((len(target_frames),) + positions.shape[1:], np.nan)
        known = np.isfinite(positions).all(axis=2).astype(float)                       # (n, anchors)
        values = np.where(known[..., None] > 0, positions, 0.0)
        # Targets are handled in chunks, each against only the frames within the window of it
        for start in range(0, len(target_frames), 200):
            targets = target_frames[start:start + 200]
            lo = np.searchsorted(frame_nums, targets[0] - self.smoothing_half_window)
            hi = np.searchsorted(frame_nums, targets[-1] + self.smoothing_half_window, side='right')
            if lo == hi:
                continue
            dt = (frame_nums[lo:hi][None, :] - targets[:, None]).astype(float)         # (m, n)
            w = self.time_kernel(dt) * weights[None, lo:hi]
            s0, s1, s2 = w @ known[lo:hi], (w * dt) @ known[lo:hi], (w * dt ** 2) @ known[lo:hi]
            count = (w > 0).astype(float) @ known[lo:hi]
            y0 = np.einsum('mn,nak->mak', w, values[lo:hi])
            y1 = np.einsum('mn,nak->mak', w * dt, values[lo:hi])
            determinant = s0 * s2 - s1 ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                line = (s2[..., None] * y0 - s1[..., None] * y1) / determinant[..., None]
                mean = y0 / s0[..., None]
            use_line = (count >= 3) & (determinant > 1e-6 * s0 ** 2)
            chunk = np.where(use_line[..., None], line, mean)
            chunk[s0 == 0] = np.nan
            fitted[start:start + len(targets)] = chunk
        return fitted

    def fit_to_anchor_points(self, image_points, frame_width, frame_height):
        """Frame to tactical view homography from the image positions of the anchor points near the frame, or None."""
        use = self.near_frame(image_points, frame_width, frame_height)
        if use.sum() < 4:
            return None
        # The anchors used must not lie (almost) on one line: their spread across the main
        # direction (root mean square, from the smallest singular value) must be large enough
        anchors = self.anchor_points[use]
        smallest_spread = np.linalg.svd(anchors - anchors.mean(axis=0), compute_uv=False)[-1] / np.sqrt(len(anchors))
        if smallest_spread < self.meters_to_pixels(self.collinear_tolerance_in_meters):
            return None
        homography, _ = cv2.findHomography(image_points[use].astype(np.float32), self.anchor_points[use])
        return homography

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

    def meters_to_tactical(self, points):
        """
        Converts court coordinates in meters (a point or an array of points) to tactical view pixels.
        """
        scale = np.array([self.width / self.actual_width_in_meters, self.height / self.actual_height_in_meters])
        return np.asarray(points, dtype=np.float32) * scale

    def tactical_to_meters(self, points):
        """
        Converts tactical view pixels (a point or an array of points) to court coordinates in meters.
        """
        scale = np.array([self.actual_width_in_meters / self.width, self.actual_height_in_meters / self.height])
        return np.asarray(points, dtype=np.float32) * scale
