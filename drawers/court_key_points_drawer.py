import numpy as np
import supervision as sv

class CourtKeypointDrawer:
    """
    A drawer class responsible for drawing court keypoints on a sequence of frames.

    Attributes:
        keypoint_color (str): Hex color value for the keypoints.
    """
    def __init__(self):
        self.keypoint_color = '#ff2c2c'

    def draw(self, frames, court_keypoints):
        """
        Draws court keypoints on a given list of frames.

        Args:
            frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            court_keypoints (list): A corresponding list of lists where each sub-list contains
                the (x, y) coordinates of court keypoints for that frame.

        Returns:
            list: A list of frames with keypoints drawn on them.
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
        
        output_frames = []
        for index,frame in enumerate(frames):
            annotated_frame = frame.copy()

            # Convert PyTorch tensor to numpy array
            keypoints_numpy = court_keypoints[index].cpu().numpy()
            if len(keypoints_numpy.xy) == 0:
                output_frames.append(annotated_frame)
                continue

            # Undetected keypoints are stored as (0, 0); only draw the detected ones, labelled
            # with their original index
            xy = keypoints_numpy.xy[0]
            detected = (xy[:, 0] > 0) & (xy[:, 1] > 0)
            if not detected.any():
                output_frames.append(annotated_frame)
                continue
            keypoints = sv.KeyPoints(xy=xy[detected][np.newaxis].astype(np.float32))
            labels = [str(i) for i in np.flatnonzero(detected)]

            # Draw dots
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints)
            # Draw labels
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints,
                labels=labels)

            output_frames.append(annotated_frame)

        return output_frames