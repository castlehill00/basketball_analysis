import cv2
import numpy as np

class CourtLinesDrawer:
    """
    A drawer class that projects the court markings back onto each frame with the frame's
    homography. When the homography is right, the drawn lines sit on the real court lines.

    Attributes:
        line_color (tuple): BGR color of the drawn court lines.
        thickness (int): Line thickness in pixels.
    """
    def __init__(self, line_color=(0, 255, 255), thickness=2):
        self.line_color = line_color
        self.thickness = thickness

    def draw(self, video_frames, homographies, court_polylines):
        """
        Draws the court markings on a sequence of frames.

        Args:
            video_frames (list): List of video frames to draw on.
            homographies (list): Frame to tactical view homography (or None) for each frame.
            court_polylines (list): Court markings as arrays of tactical view points.

        Yields:
            numpy.ndarray: Each frame with the court markings drawn on it.
        """
        for frame, homography in zip(video_frames, homographies):
            frame = frame.copy()

            if homography is not None:
                tactical_to_frame = np.linalg.inv(homography)
                for polyline in court_polylines:
                    points = cv2.perspectiveTransform(np.asarray(polyline, dtype=np.float32).reshape(-1, 1, 2),
                                                      tactical_to_frame).reshape(-1, 2)

                    # A badly wrong homography can throw points absurdly far away
                    if not np.all(np.isfinite(points)) or np.abs(points).max() > 1e5:
                        continue

                    cv2.polylines(frame, [np.round(points).astype(np.int32)], False, self.line_color, self.thickness)

            yield frame
