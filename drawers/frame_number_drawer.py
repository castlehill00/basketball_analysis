import cv2
class FrameNumberDrawer:
    def __init__(self):
        pass

    def draw(self,frames):
        # Write the frame number on the top left corner of the frame
        for i, frame in enumerate(frames):
            frame = frame.copy()
            cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            yield frame
