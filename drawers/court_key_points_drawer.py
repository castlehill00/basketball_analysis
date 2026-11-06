import cv2
import numpy as np
import supervision as sv

class CourtKeypointDrawer:
    """
    A drawer class responsible for drawing court keypoints on a sequence of frames.

    Only the keypoints that agree with the frame's court homography are drawn. The keypoint
    model also outputs keypoints in views that show no court (close-ups, the cameras behind the
    baskets) and puts some keypoints far from where they are; drawn, those jump around and make
    the court look wrong even where its lines are right.

    Attributes:
        keypoint_color (str): Hex color value for the keypoints.
        max_distance_pixels (float): A detected keypoint is drawn only if it lies within this
            distance of where the homography puts it (on the 2023 World Cup final, 90% of the
            keypoints in frames with a court are within 20 px and nearly all others hundreds
            of pixels away).
    """
    def __init__(self, max_distance_pixels=20):
        self.keypoint_color = '#ff2c2c'
        self.max_distance_pixels = max_distance_pixels

    def draw(self, frames, court_keypoints, homographies, tactical_view_converter):
        """
        Draws court keypoints on a given list of frames.

        Args:
            frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            court_keypoints (list): Court keypoints (ultralytics Keypoints) for each frame.
            homographies (list): Frame to tactical view homography (or None) for each frame;
                frames without one get no keypoints.
            tactical_view_converter (TacticalViewConverter): Gives where the keypoints are in the
                tactical view and which keypoint is the same one on the other half of the court.

        Yields:
            numpy.ndarray: Each frame with its keypoints drawn on it.
        """
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8)

        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

        court_points = np.array(tactical_view_converter.key_points, dtype=np.float32).reshape(-1, 1, 2)
        mirrored_indices = np.array(tactical_view_converter.mirrored_key_point_indices)

        for index,frame in enumerate(frames):
            annotated_frame = frame.copy()

            # Convert PyTorch tensor to numpy array
            keypoints_numpy = court_keypoints[index].cpu().numpy()
            homography = homographies[index]
            if homography is None or len(keypoints_numpy.xy) == 0:
                yield annotated_frame
                continue

            # Where the homography puts each keypoint in the frame
            try:
                expected = cv2.perspectiveTransform(court_points, np.linalg.inv(homography)).reshape(-1, 2)
            except np.linalg.LinAlgError:
                yield annotated_frame
                continue

            # Undetected keypoints are stored as (0, 0). The model sometimes gives the visible half
            # of the court the labels of the other half (see TacticalViewConverter), so a keypoint
            # also agrees if it lies where its counterpart on the other half is put
            xy = keypoints_numpy.xy[0]
            detected = (xy[:, 0] > 0) & (xy[:, 1] > 0)
            with np.errstate(invalid='ignore'):
                distance = np.minimum(np.linalg.norm(xy - expected, axis=1),
                                      np.linalg.norm(xy - expected[mirrored_indices], axis=1))
                shown = detected & (distance <= self.max_distance_pixels)
            if not shown.any():
                yield annotated_frame
                continue
            # Labelled with their original index
            keypoints = sv.KeyPoints(xy=xy[shown][np.newaxis].astype(np.float32))
            labels = [str(i) for i in np.flatnonzero(shown)]

            # Draw dots
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints)
            # Draw labels
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints,
                labels=labels)

            yield annotated_frame
