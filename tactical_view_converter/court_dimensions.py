"""
Court dimensions for the leagues the tactical view supports.

All distances are in metres. Court coordinates have x along the length of the court (0 at the
left baseline) and y across it (0 at the far sideline, as seen from the main camera).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CourtDimensions:
    length: float
    width: float
    # Floor point under the centre of the basket, measured from the baseline
    basket_from_baseline: float
    three_point_radius: float
    # The straight corner three-point lines run parallel to the sidelines at this distance
    corner_three_from_sideline: float
    lane_width: float
    free_throw_from_baseline: float
    center_circle_radius: float

    @property
    def corner_three_depth(self):
        """Distance from the baseline where the straight corner three-point line meets the arc."""
        lateral = self.width / 2 - self.corner_three_from_sideline
        return self.basket_from_baseline + np.sqrt(self.three_point_radius ** 2 - lateral ** 2)

    def key_points(self):
        """The 18 court keypoints predicted by the court keypoint model, in court coordinates."""
        length, width = self.length, self.width
        corner = self.corner_three_from_sideline
        lane_near, lane_far = width / 2 - self.lane_width / 2, width / 2 + self.lane_width / 2
        free_throw = self.free_throw_from_baseline
        return [
            # left edge
            (0, 0), (0, corner), (0, lane_near), (0, lane_far), (0, width - corner), (0, width),
            # middle line
            (length / 2, width), (length / 2, 0),
            # left free throw line
            (free_throw, lane_near), (free_throw, lane_far),
            # right edge
            (length, width), (length, width - corner), (length, lane_far), (length, lane_near),
            (length, corner), (length, 0),
            # right free throw line
            (length - free_throw, lane_near), (length - free_throw, lane_far),
        ]

    def lines(self, arc_points=40):
        """Court markings as polylines (arrays of court coordinates), used for drawing."""
        length, width = self.length, self.width
        lane_near, lane_far = width / 2 - self.lane_width / 2, width / 2 + self.lane_width / 2
        free_throw = self.free_throw_from_baseline

        angles = np.linspace(0, 2 * np.pi, arc_points)
        center_circle = np.stack([length / 2 + self.center_circle_radius * np.cos(angles),
                                  width / 2 + self.center_circle_radius * np.sin(angles)], axis=1)
        # The free-throw circle is centred on the free-throw line and as large as the centre circle
        left_free_throw_circle = np.stack([free_throw + self.center_circle_radius * np.cos(angles),
                                           width / 2 + self.center_circle_radius * np.sin(angles)], axis=1)

        # Left three-point line: corner line, arc around the basket, other corner line
        lateral = width / 2 - self.corner_three_from_sideline
        arc_end = np.arctan2(lateral, self.corner_three_depth - self.basket_from_baseline)
        arc_angles = np.linspace(-arc_end, arc_end, arc_points)
        arc = np.stack([self.basket_from_baseline + self.three_point_radius * np.cos(arc_angles),
                        width / 2 + self.three_point_radius * np.sin(arc_angles)], axis=1)
        left_three = np.vstack([[0, width / 2 - lateral], arc, [0, width / 2 + lateral]])
        left_lane = np.array([[0, lane_near], [free_throw, lane_near], [free_throw, lane_far], [0, lane_far]])

        def mirrored(polyline):
            return np.stack([length - polyline[:, 0], polyline[:, 1]], axis=1)

        return [
            np.array([[0, 0], [length, 0], [length, width], [0, width], [0, 0]]),
            np.array([[length / 2, 0], [length / 2, width]]),
            center_circle,
            left_lane, mirrored(left_lane),
            left_three, mirrored(left_three),
            left_free_throw_circle, mirrored(left_free_throw_circle),
        ]

    def three_point_margin(self, court_position):
        """
        Signed distance from the three-point line of the basket on the same half of the court.

        Positive means beyond the line (a three-pointer). A shot taken with a foot on the line is
        a two-pointer, so only a positive margin counts as a three.
        """
        x, y = court_position
        distance_from_baseline = min(x, self.length - x)
        lateral_offset = abs(y - self.width / 2)
        if distance_from_baseline <= self.corner_three_depth:
            return lateral_offset - (self.width / 2 - self.corner_three_from_sideline)
        return np.hypot(distance_from_baseline - self.basket_from_baseline, lateral_offset) - self.three_point_radius


# FIBA (also used by the CBA) and NBA rule books
FIBA = CourtDimensions(length=28.0, width=15.0, basket_from_baseline=1.575, three_point_radius=6.75,
                       corner_three_from_sideline=0.9, lane_width=4.9, free_throw_from_baseline=5.8,
                       center_circle_radius=1.8)
NBA = CourtDimensions(length=28.65, width=15.24, basket_from_baseline=1.6, three_point_radius=7.24,
                      corner_three_from_sideline=0.914, lane_width=4.877, free_throw_from_baseline=5.791,
                      center_circle_radius=1.83)
COURT_DIMENSIONS = {'fiba': FIBA, 'nba': NBA}
