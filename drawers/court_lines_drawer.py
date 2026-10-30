import cv2
import numpy as np

class CourtLinesDrawer:
    """
    A drawer class that projects the court template back onto each frame with the frame's
    homography. When the homography is right, the drawn lines sit on the real court lines.

    Attributes:
        line_color (tuple): BGR color of the drawn court lines.
        thickness (int): Line thickness in pixels.
    """
    def __init__(self, line_color=(0, 255, 255), thickness=2):
        self.line_color = line_color
        self.thickness = thickness

    def draw(self, video_frames, homographies, tactical_court_keypoints, court_lines):
        """
        Draws the court lines on a sequence of frames.

        Args:
            video_frames (list): List of video frames to draw on.
            homographies (list): Frame to tactical view homography (or None) for each frame.
            tactical_court_keypoints (list): Court keypoints in tactical view coordinates.
            court_lines (list): Pairs of keypoint indices that form the court lines.

        Returns:
            list: List of frames with the court lines drawn on them.
        """
        output_video_frames = []
        for frame, homography in zip(video_frames, homographies):
            frame = frame.copy()

            if homography is not None:
                tactical_to_frame = np.linalg.inv(homography)
                for start_index, end_index in court_lines:
                    line = np.array([[tactical_court_keypoints[start_index]],
                                     [tactical_court_keypoints[end_index]]], dtype=np.float32)
                    line = cv2.perspectiveTransform(line, tactical_to_frame).reshape(2, 2)

                    # A badly wrong homography can throw points absurdly far away
                    if not np.all(np.isfinite(line)) or np.abs(line).max() > 1e5:
                        continue

                    (x1, y1), (x2, y2) = line
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.line_color, self.thickness)

            output_video_frames.append(frame)

        return output_video_frames
