import cv2
import numpy as np

class ShotDrawer:
    """
    A drawer class that marks detected shots: the take-off spot on the court labelled with the
    points (2PT / 3PT), and a running count of shots in the corner of the frame.

    Attributes:
        colors (dict): BGR color for two- and three-point shots.
        display_frames (int): How long a shot stays marked after the ball reached the hoop.
    """
    def __init__(self, display_frames=45):
        self.colors = {2: (255, 200, 0), 3: (0, 140, 255)}
        self.display_frames = display_frames

    def draw(self, video_frames, shots, homographies, tactical_view_converter):
        """
        Draws the shots on a sequence of frames.

        Args:
            video_frames (list): List of video frames to draw on.
            shots (list): Shots from ShotDetector.detect_shots.
            homographies (list): Frame to tactical view homography (or None) for each frame.
            tactical_view_converter (TacticalViewConverter): Converts court meters to tactical pixels.

        Yields:
            numpy.ndarray: Each frame with the shots drawn on it.
        """
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            for shot in shots:
                if shot['points'] is None or not shot['take_off_frame'] <= frame_num <= shot['end_frame'] + self.display_frames:
                    continue
                if homographies[frame_num] is None:
                    continue
                # Project the take-off spot from the court into this frame
                tactical = tactical_view_converter.meters_to_tactical(shot['court_position']).reshape(1, 1, 2)
                x, y = cv2.perspectiveTransform(tactical, np.linalg.inv(homographies[frame_num]))[0][0]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                color = self.colors[shot['points']]
                cv2.circle(frame, (int(x), int(y)), 10, color, -1)
                cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 0), 2)
                label = f"{shot['points']}PT ({shot['three_point_margin']:+.1f} m)"
                cv2.putText(frame, label, (int(x) + 14, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(frame, label, (int(x) + 14, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Running count of shots whose ball has reached the hoop
            counts = {2: 0, 3: 0}
            for shot in shots:
                if shot['points'] is not None and shot['hoop_frame'] <= frame_num:
                    counts[shot['points']] += 1
            text = f"Shots  2PT: {counts[2]}  3PT: {counts[3]}"
            x = frame.shape[1] - 330
            cv2.putText(frame, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            yield frame
