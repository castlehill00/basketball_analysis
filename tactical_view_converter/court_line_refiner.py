"""
Refines the court homography of each frame with the painted court lines found in the frame.

A homography fitted to the court keypoints is only reliable near them. When the keypoints are
bunched in one part of the frame (often just the lane and the baseline near one basket), the
rest of the court is extrapolated, and the half-court line or the far sideline can end up a
hundred pixels off; the keypoints themselves are also often a few metres-worth of pixels off.
The painted lines (sidelines, half-court line, centre circle, lanes, three-point lines,
free-throw circles) cover the whole visible court, so the homography is adjusted until the
projected court lines lie on the painted ones.
"""
import hashlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.optimize import least_squares

# Pixel sizes are given for 720p frames and scaled with the frame height
REFERENCE_HEIGHT = 720


def line_mask(frame, bottom_margin=0.15):
    """
    Finds the painted lines on a wooden floor: thin structures that are brighter than their
    surroundings, unsaturated (white) and lie on orange-brown wood.

    Args:
        frame (numpy.ndarray): BGR frame.
        bottom_margin (float): Share of the frame height at the bottom that is ignored (the score
            bug of most broadcasts covers the court there).

    Returns:
        tuple: (mask of line pixels as uint8, share of the frame above the margin that is wood).
    """
    height = frame.shape[0]
    size = int(round(11 * height / REFERENCE_HEIGHT)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    tophat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    # The floor next to a line (with the line removed by the opening) is orange-brown wood
    background = cv2.cvtColor(cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel), cv2.COLOR_BGR2HSV)
    wood = (background[:, :, 0] >= 5) & (background[:, :, 0] <= 18) & (background[:, :, 1] >= 30) & \
        (background[:, :, 1] <= 130) & (background[:, :, 2] >= 110)
    top = int(round(height * (1 - bottom_margin)))
    floor_share = float(wood[:top].mean())
    wood = cv2.dilate(wood.astype(np.uint8), kernel) > 0
    mask = ((tophat > 30) & (hsv[:, :, 1] < 70) & (value > 150) & wood).astype(np.uint8)
    mask[top:] = 0
    # Drop small blobs (shoes, socks, the ball's highlight)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.maximum(stats[:, 2], stats[:, 3]) >= 15 * height / REFERENCE_HEIGHT
    keep[0] = False
    return keep[components].astype(np.uint8), floor_share


def sample_polylines(polylines, step):
    """Points every step along the polylines."""
    points = []
    for polyline in polylines:
        polyline = np.asarray(polyline, dtype=float)
        for start, end in zip(polyline, polyline[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / step)))
            points.append(start + (end - start) * np.linspace(0, 1, count, endpoint=False)[:, None])
    return np.vstack(points)


def project(court_to_frame, points):
    """Projects points (n, 2); returns their image positions and the projective scale (> 0 in front of the camera)."""
    projected = np.c_[points, np.ones(len(points))] @ court_to_frame.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return projected[:, :2] / projected[:, 2:3], projected[:, 2]


def bilinear(image, x, y):
    height, width = image.shape
    x = np.clip(x, 0, width - 1.001)
    y = np.clip(y, 0, height - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy) +
            image[y0 + 1, x0] * (1 - fx) * fy + image[y0 + 1, x0 + 1] * fx * fy)


def in_view(positions, scales, width, top):
    with np.errstate(invalid='ignore'):
        return (scales > 0) & np.isfinite(positions).all(axis=1) & (positions[:, 0] >= 0) & \
            (positions[:, 0] <= width - 1) & (positions[:, 1] >= 0) & (positions[:, 1] < top)


def line_support(distance_to_line, court_to_frame, line_points, top, match_pixels):
    """Share of the court line points in view that lie within match_pixels of a painted line (nan if too few are in view)."""
    positions, scales = project(court_to_frame, line_points)
    visible = in_view(positions, scales, distance_to_line.shape[1], top)
    if visible.sum() < 20:
        return np.nan
    distances = bilinear(distance_to_line, positions[visible, 0], positions[visible, 1])
    return float(np.mean(distances <= match_pixels))


def optimize(distance_to_line, court_to_frame, line_points, keypoint_positions, keypoint_court_points,
             keypoint_sigma, scales, max_evaluations):
    """
    Adjusts the court-to-frame homography so that the projected line points lie on painted lines.

    The 8 free entries of the homography (in normalised coordinates) are fitted by robust least
    squares. Each line point in view contributes its distance to the nearest painted line pixel
    (from the distance transform), with a Cauchy loss so hidden or undetected lines do not pull;
    the fit is done twice, first with a wide loss scale to come close, then with a narrow one.
    A point that leaves the frame keeps a growing residual, so moving lines out of view is not a
    way to escape them. The keypoints that agreed with the start enter with a quadratic penalty:
    the fit may move away from them by about keypoint_sigma, but cannot collapse the court onto a
    single painted line, which would also put many line points on a line.
    """
    height, width = distance_to_line.shape
    positions, projective_scales = project(court_to_frame, line_points)
    line_points = line_points[in_view(positions, projective_scales, width, height)]
    if len(line_points) < 20:
        return court_to_frame
    gradient_x = cv2.Sobel(distance_to_line, cv2.CV_32F, 1, 0, ksize=3) / 8
    gradient_y = cv2.Sobel(distance_to_line, cv2.CV_32F, 0, 1, ksize=3) / 8

    # Normalised coordinates keep the 8 parameters of similar size
    pixel_scale = 2.0 / width
    to_image_units = np.array([[pixel_scale, 0, -1], [0, pixel_scale, -height / width], [0, 0, 1]])
    court_center = np.mean(line_points, axis=0)
    court_scale = 1.0 / max(np.ptp(line_points, axis=0).max(), 1e-6)
    to_court_units = np.array([[court_scale, 0, -court_center[0] * court_scale],
                               [0, court_scale, -court_center[1] * court_scale], [0, 0, 1]])
    normalised = to_image_units @ court_to_frame @ np.linalg.inv(to_court_units)
    params = (normalised / normalised[2, 2]).ravel()[:8]

    def homogeneous(points):
        return np.c_[(points - court_center) * court_scale, np.ones(len(points))]

    line_x = homogeneous(line_points)
    keypoint_x = homogeneous(keypoint_court_points)
    line_count = len(line_points)

    def image_positions(p, x):
        denominator = x @ np.r_[p[6:8], 1.0]
        u = (x @ p[0:3]) / denominator
        v = (x @ p[3:6]) / denominator
        return (u + 1) / pixel_scale, (v + height / width) / pixel_scale, u, v, denominator

    def position_jacobian(u, v, denominator, x):
        ju, jv = np.zeros((len(x), 8)), np.zeros((len(x), 8))
        ju[:, 0:3] = x / denominator[:, None]
        ju[:, 6:8] = -(u / denominator)[:, None] * x[:, :2]
        jv[:, 3:6] = x / denominator[:, None]
        jv[:, 6:8] = -(v / denominator)[:, None] * x[:, :2]
        return ju / pixel_scale, jv / pixel_scale

    def residuals(p, loss_scale):
        px, py, _, _, denominator = image_positions(p, line_x)
        cx, cy = np.clip(px, 0, width - 1.001), np.clip(py, 0, height - 1.001)
        r = bilinear(distance_to_line, np.nan_to_num(cx), np.nan_to_num(cy)) + np.abs(px - cx) + np.abs(py - cy)
        r[~(denominator > 1e-9)] = 10 * width
        kx, ky, _, _, _ = image_positions(p, keypoint_x)
        weight = loss_scale / keypoint_sigma
        return np.nan_to_num(np.r_[r, (kx - keypoint_positions[:, 0]) * weight, (ky - keypoint_positions[:, 1]) * weight],
                             nan=10 * width)

    def jacobian(p, loss_scale):
        px, py, u, v, denominator = image_positions(p, line_x)
        cx, cy = np.clip(px, 0, width - 1.001), np.clip(py, 0, height - 1.001)
        dx = np.where(px == cx, bilinear(gradient_x, np.nan_to_num(cx), np.nan_to_num(cy)), np.sign(px - cx))
        dy = np.where(py == cy, bilinear(gradient_y, np.nan_to_num(cx), np.nan_to_num(cy)), np.sign(py - cy))
        ju, jv = position_jacobian(u, v, denominator, line_x)
        j = dx[:, None] * ju + dy[:, None] * jv
        j[~(denominator > 1e-9)] = 0
        _, _, ku, kv, kdenominator = image_positions(p, keypoint_x)
        kju, kjv = position_jacobian(ku, kv, kdenominator, keypoint_x)
        weight = loss_scale / keypoint_sigma
        return np.nan_to_num(np.vstack([j, kju * weight, kjv * weight]), nan=0.0, posinf=0.0, neginf=0.0)

    def loss(z):
        """Cauchy for the line points, quadratic for the keypoints."""
        rho = np.empty((3, len(z)))
        line, keypoint = slice(0, line_count), slice(line_count, None)
        rho[0, line] = np.log1p(z[line])
        rho[1, line] = 1 / (1 + z[line])
        rho[2, line] = -1 / (1 + z[line]) ** 2
        rho[0, keypoint], rho[1, keypoint], rho[2, keypoint] = z[keypoint], 1.0, 0.0
        return rho

    try:
        for loss_scale in scales:
            fit = least_squares(residuals, params, jac=jacobian, args=(loss_scale,), loss=loss, f_scale=loss_scale,
                                x_scale='jac', max_nfev=max_evaluations)
            params = fit.x
    except (ValueError, np.linalg.LinAlgError):
        return court_to_frame
    refined = np.linalg.inv(to_image_units) @ np.append(params, 1.0).reshape(3, 3) @ to_court_units
    return refined / refined[2, 2]


def refine_frame(frame, frame_to_tactical, keypoints, settings):
    """
    Refines one frame's homography. See CourtLineRefiner for the rules.

    Args:
        frame (numpy.ndarray): BGR frame.
        frame_to_tactical (numpy.ndarray): Homography from the frame to the tactical view.
        keypoints (numpy.ndarray): Detected court keypoints (18, 2), (0, 0) where undetected.
        settings (dict): From CourtLineRefiner.settings().

    Returns:
        tuple: (refined frame-to-tactical homography, or None when the frame shows too little
            floor; whether the line fit was used).
    """
    height, width = frame.shape[:2]
    scale = height / REFERENCE_HEIGHT
    mask, floor_share = line_mask(frame, settings['bottom_margin'])
    if floor_share < settings['min_floor_share']:
        return None, False
    top = int(round(height * (1 - settings['bottom_margin'])))
    distance_to_line = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)

    tactical_to_meters = np.diag(settings['meters_per_tactical_pixel'] + [1.0])
    try:
        court_to_frame = np.linalg.inv(tactical_to_meters @ frame_to_tactical)
    except np.linalg.LinAlgError:
        return frame_to_tactical, False
    court_to_frame /= court_to_frame[2, 2]

    # The keypoints that agree with the start, under their own label or the other half's
    court_keypoints = np.asarray(settings['court_keypoints'])
    mirrored = np.asarray(settings['mirrored_keypoint_indices'])
    expected, _ = project(court_to_frame, court_keypoints)
    detected = (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)
    with np.errstate(invalid='ignore'):
        own = np.linalg.norm(keypoints - expected, axis=1)
        other = np.linalg.norm(keypoints - expected[mirrored], axis=1)
        agree = detected & (np.minimum(own, other) <= settings['keypoint_match_pixels'] * scale)
    keypoint_court_points = np.where((own <= other)[:, None], court_keypoints, court_keypoints[mirrored])[agree]
    keypoint_positions = keypoints[agree]

    line_points = np.asarray(settings['line_points'])
    refined = optimize(distance_to_line, court_to_frame, line_points, keypoint_positions, keypoint_court_points,
                       settings['keypoint_sigma_pixels'] * scale,
                       [s * scale for s in settings['loss_scales_pixels']], settings['max_evaluations'])

    # Use the fit only if it brings the lines closer, keeps the keypoints and does not squash the court
    match = settings['match_pixels'] * scale
    before = line_support(distance_to_line, court_to_frame, line_points, top, match)
    after = line_support(distance_to_line, refined, line_points, top, match)
    if not (after >= before + settings['min_support_gain']):
        return frame_to_tactical, False
    if len(keypoint_positions):
        moved, _ = project(refined, keypoint_court_points)
        if np.median(np.linalg.norm(moved - keypoint_positions, axis=1)) > settings['max_keypoint_error_pixels'] * scale:
            return frame_to_tactical, False
    start_positions, start_scales = project(court_to_frame, line_points)
    visible = in_view(start_positions, start_scales, width, top)
    end_positions, end_scales = project(refined, line_points[visible])
    if not np.all(end_scales > 0):
        return frame_to_tactical, False
    area_before = cv2.contourArea(cv2.convexHull(start_positions[visible].astype(np.float32)))
    area_after = cv2.contourArea(cv2.convexHull(end_positions.astype(np.float32)))
    if not (settings['area_ratio_range'][0] <= area_after / max(area_before, 1e-6) <= settings['area_ratio_range'][1]):
        return frame_to_tactical, False

    frame_to_tactical_refined = np.linalg.inv(tactical_to_meters) @ np.linalg.inv(refined)
    return frame_to_tactical_refined / frame_to_tactical_refined[2, 2], True


def refine_chunk(video_path, frame_nums, homographies, keypoints, settings):
    """Refines the homographies of consecutive frames of a video file (run in a worker process)."""
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_nums[0])
    results = []
    frame_num = frame_nums[0]
    for target, homography, frame_keypoints in zip(frame_nums, homographies, keypoints):
        while frame_num < target:
            capture.grab()
            frame_num += 1
        ok, frame = capture.read()
        frame_num += 1
        results.append(refine_frame(frame, homography, frame_keypoints, settings) if ok else (homography, False))
    capture.release()
    return results


class CourtLineRefiner:
    """
    Refines the per-frame court homographies with the painted lines (see the module docstring).

    For each frame with a homography, the line fit is used only if it brings the projected
    lines closer to the painted ones, keeps the agreeing keypoints within max_keypoint_error_pixels
    and does not shrink or grow the visible court by more than area_ratio_range; otherwise the
    keypoint homography is kept. Without painted lines to go by (as on floors where they are not
    found) nothing changes. Frames in which less than min_floor_share of the picture is wooden
    floor lose their homography: they are close-ups, where the keypoint model still finds
    keypoints on faces and shirts.
    """
    def __init__(self, tactical_view_converter,
                 line_step_in_meters=0.5,
                 keypoint_sigma_pixels=10,
                 max_keypoint_error_pixels=30,
                 min_support_gain=0.02,
                 min_floor_share=0.04,
                 workers=4):
        """
        Args:
            tactical_view_converter (TacticalViewConverter): Gives the court dimensions and keypoints.
            line_step_in_meters (float): Spacing of the points sampled along the court lines.
            keypoint_sigma_pixels (float): How far (720p pixels) the fit may move the keypoints
                that agreed with the start before that starts to cost as much as a line point.
            max_keypoint_error_pixels (float): The fit is not used if those keypoints end up
                further (median, 720p pixels) from their detected positions.
            min_support_gain (float): The share of court line points on painted lines must grow
                by at least this much.
            min_floor_share (float): Frames with less wooden floor than this share lose their homography.
            workers (int): Processes used when the frames come from a video file.
        """
        self.tactical_view_converter = tactical_view_converter
        self.line_step_in_meters = line_step_in_meters
        self.keypoint_sigma_pixels = keypoint_sigma_pixels
        self.max_keypoint_error_pixels = max_keypoint_error_pixels
        self.min_support_gain = min_support_gain
        self.min_floor_share = min_floor_share
        self.workers = workers

    def settings(self):
        converter = self.tactical_view_converter
        return {
            'line_points': sample_polylines(converter.court.lines(), self.line_step_in_meters).tolist(),
            'court_keypoints': [list(point) for point in converter.court.key_points()],
            'mirrored_keypoint_indices': list(converter.mirrored_key_point_indices),
            'meters_per_tactical_pixel': [converter.actual_width_in_meters / converter.width,
                                          converter.actual_height_in_meters / converter.height],
            'bottom_margin': 0.15,
            'keypoint_match_pixels': 20,
            'keypoint_sigma_pixels': self.keypoint_sigma_pixels,
            'max_keypoint_error_pixels': self.max_keypoint_error_pixels,
            'min_support_gain': self.min_support_gain,
            'min_floor_share': self.min_floor_share,
            'match_pixels': 4,
            'loss_scales_pixels': [15.0, 3.0],
            'max_evaluations': 20,
            'area_ratio_range': [0.6, 1.6],
        }

    def get_homographies(self, frames, keypoints_list, camera_shots, read_from_stub=False, stub_path=None):
        """
        The court homography of every frame. It is fitted to the keypoints and checked and
        smoothed within each camera shot (TacticalViewConverter.get_homographies), which decides
        which frames show the court; then adjusted to the painted lines frame by frame
        (refine_homographies); then smoothed again within each shot, without the checks, to
        remove the frame-to-frame noise of the adjustment.

        Args:
            frames (sequence): Video frames.
            keypoints_list (list): Court keypoints (ultralytics Keypoints) per frame.
            camera_shots (list or None): (first frame, last frame) of each camera shot.
            read_from_stub (bool): Whether to reuse the cached adjustment.
            stub_path (str): Path of the cache file.

        Returns:
            list: A 3x3 frame-to-tactical homography (or None) per frame.
        """
        converter = self.tactical_view_converter
        checked = converter.get_homographies(keypoints_list, camera_shots)
        adjusted = self.refine_homographies(frames, checked, keypoints_list, read_from_stub, stub_path)
        return converter.get_homographies(keypoints_list, camera_shots, adjusted, check_shots=False)

    def refine_homographies(self, frames, homographies, keypoints_list, read_from_stub=False, stub_path=None):
        """
        Refines the homography of every frame that has one.

        Args:
            frames (sequence): Video frames (utils.VideoFrames is decoded in worker processes).
            homographies (list): Frame-to-tactical homography (or None) per frame.
            keypoints_list (list): Court keypoints (ultralytics Keypoints) per frame.
            read_from_stub (bool): Whether to reuse a cached result for the same input and settings.
            stub_path (str): Path of the cache file.

        Returns:
            list: The refined homography (or None) per frame.
        """
        from utils import read_stub, save_stub

        settings = self.settings()
        targets = [t for t, homography in enumerate(homographies) if homography is not None]
        # The cache is only valid for the same homographies and settings
        digest = hashlib.sha1(repr(sorted((k, v) for k, v in settings.items() if k != 'line_points')).encode())
        for t in targets:
            digest.update(np.round(np.asarray(homographies[t], dtype=float), 6).tobytes())
        fingerprint = digest.hexdigest()
        cached = read_stub(read_from_stub, stub_path)
        if cached is not None and cached.get('fingerprint') == fingerprint:
            return cached['homographies']

        keypoints = []
        for t in targets:
            xy = keypoints_list[t].xy
            xy = xy.cpu().numpy() if hasattr(xy, 'cpu') else np.asarray(xy)
            keypoints.append(xy[0] if len(xy) else np.zeros((len(settings['court_keypoints']), 2)))

        refined = [None] * len(homographies)
        video_path = getattr(frames, 'video_path', None)
        if video_path is not None and self.workers > 1 and targets:
            chunk_size = max(1, int(np.ceil(len(targets) / (self.workers * 8))))
            chunks = [list(range(i, min(i + chunk_size, len(targets)))) for i in range(0, len(targets), chunk_size)]
            with ProcessPoolExecutor(self.workers) as pool:
                futures = [pool.submit(refine_chunk, video_path, [targets[i] for i in chunk],
                                       [homographies[targets[i]] for i in chunk], [keypoints[i] for i in chunk], settings)
                           for chunk in chunks]
                for chunk, future in zip(chunks, futures):
                    for i, (homography, _) in zip(chunk, future.result()):
                        refined[targets[i]] = homography
        else:
            for i, t in enumerate(targets):
                refined[t] = refine_frame(frames[t], homographies[t], keypoints[i], settings)[0]

        save_stub(stub_path, {'fingerprint': fingerprint, 'homographies': refined})
        return refined
