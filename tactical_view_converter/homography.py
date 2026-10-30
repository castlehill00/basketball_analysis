import numpy as np
import cv2 

class Homography:
    def __init__(self, source: np.ndarray, target: np.ndarray, ransac_threshold: float = None) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        if ransac_threshold is None:
            self.m, mask = cv2.findHomography(source, target)
        else:
            # RANSAC leaves out points that disagree with the rest, e.g. a misplaced keypoint
            self.m, mask = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

        # Which source points the homography was fitted to (all of them without RANSAC)
        self.inliers = np.ones(len(source), dtype=bool) if mask is None else mask.ravel().astype(bool)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")
        
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)





