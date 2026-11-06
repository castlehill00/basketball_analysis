import sys
from collections import Counter

import numpy as np
import cv2

sys.path.append('../')
from utils import get_center_of_bbox, get_foot_position


class ShotDetector:
    """
    Detects shot attempts and classifies them as two- or three-pointers.

    A shot is found when the ball enters the area around the hoop. Going back from there, the
    shooter is the last player of the team in possession who held the ball for a few frames.
    Whether the shot is worth two or three points depends on where the shooter's feet were at
    take-off, which is mapped onto the court with the frame's homography and compared with the
    three-point line.
    """
    def __init__(self, tactical_view_converter):
        """
        Args:
            tactical_view_converter (TacticalViewConverter): Provides the court dimensions and the
                conversion from tactical view pixels to meters.
        """
        self.tactical_view_converter = tactical_view_converter
        self.court = tactical_view_converter.court

        # Area around the hoop, in hoop widths from the hoop centre (negative y is above it). It
        # reaches well above the rim: on long shots the ball is often last seen there, before the
        # backboard or motion blur hides it
        self.hoop_zone_half_width = 1.5
        self.hoop_zone_top = -4.0
        self.hoop_zone_bottom = 1.0
        # Ball positions are interpolated over gaps up to this many frames, no further
        self.max_ball_gap = 6
        # Hoop detections are reused for up to this many frames when the hoop is missed
        self.max_hoop_gap = 15
        # Near-hoop frames closer together than this belong to the same shot
        self.max_episode_gap = 20
        # How far back from the hoop the shooter is searched for
        self.max_flight_frames = 60
        # Frames in a row the shooter must hold the ball, so a ball flying past a player is ignored
        self.min_hold_frames = 3
        # Margin around a player's box within which the ball counts as held
        self.hold_margin = 0.2
        # Take-off is searched this many frames before the release (the feet are on the floor
        # just before the jump; at the release the shooter is in the air)
        self.take_off_window = (15, 4)
        # Use the release frame instead of the estimated take-off (for comparison)
        self.use_release_frame = False
        # The shooter must be on the team that had the ball before the shot: at the release a
        # defender's box often covers the ball too, and false ball detections land on any player
        self.use_teams = True
        # Frames before the ball reaches the hoop over which possession is counted
        self.possession_frames = 100
        # Leave out shots seen without the court lines (no homography): free throws, replays and
        # close-ups are shown from other cameras, and the points cannot be told anyway
        self.require_court_position = True

    def detect_shots(self, ball_tracks, hoop_tracks, player_tracks, homographies, player_teams=None):
        """
        Finds the shot attempts in a video.

        Args:
            ball_tracks (list): Ball detections per frame after remove_wrong_detections (before
                interpolation, so that long gaps are not filled in).
            hoop_tracks (list): Hoop detections per frame from BallTracker.get_tracks_of_classes.
            player_tracks (list): Player tracks per frame.
            homographies (list): Frame to tactical view homography (or None) per frame.
            player_teams (list): Team of every player per frame (from TeamAssigner), or None to
                leave out the team check.

        Returns:
            list: One dictionary per shot with the frames involved, the shooter, the take-off
                position on the court in meters, its distance from the three-point line and the
                points (2 or 3). The position is None when no homography was available (only if
                require_court_position is off).
        """
        balls = self.fill_short_gaps([self.get_center(frame.get(1)) for frame in ball_tracks], self.max_ball_gap)
        hoops = self.fill_short_gaps([frame.get(1, {}).get('bbox') for frame in hoop_tracks], self.max_hoop_gap)
        holders = [self.get_ball_holder(ball, players) for ball, players in zip(balls, player_tracks)]

        shots = []
        for first_frame, last_frame in self.find_hoop_episodes(balls, hoops):
            team = None
            if self.use_teams and player_teams is not None:
                team = self.find_team_in_possession(holders, player_teams, first_frame)
            release_frame, shooter_id = self.find_release(balls, player_tracks, first_frame, player_teams, team)
            # Two episodes of one shot (the ball left the area and came back) lead to the same release
            if shooter_id is None or (shots and shots[-1]['release_frame'] == release_frame):
                continue

            take_off_frame = release_frame if self.use_release_frame else \
                self.find_take_off(player_tracks, shooter_id, release_frame)
            position = self.get_court_position(player_tracks, homographies, shooter_id, take_off_frame)
            if position is None and self.require_court_position:
                continue
            margin = None if position is None else float(self.court.three_point_margin(position))

            shots.append({
                'release_frame': release_frame,
                'take_off_frame': take_off_frame,
                'hoop_frame': first_frame,
                'end_frame': last_frame,
                'shooter_id': int(shooter_id),
                'court_position': None if position is None else [float(v) for v in position],
                'three_point_margin': margin,
                'points': None if margin is None else (3 if margin > 0 else 2),
            })
        return shots

    def find_hoop_episodes(self, balls, hoops):
        """
        Groups the frames where the ball is in the area around the hoop into episodes.

        Returns:
            list: (first frame, last frame) of each episode.
        """
        near_frames = []
        for frame_num, (ball, hoop) in enumerate(zip(balls, hoops)):
            if ball is None or hoop is None:
                continue
            hoop_x, hoop_y = (hoop[0] + hoop[2]) / 2, (hoop[1] + hoop[3]) / 2
            hoop_width = hoop[2] - hoop[0]
            dx, dy = (ball[0] - hoop_x) / hoop_width, (ball[1] - hoop_y) / hoop_width
            if abs(dx) <= self.hoop_zone_half_width and self.hoop_zone_top <= dy <= self.hoop_zone_bottom:
                near_frames.append(frame_num)

        episodes = []
        for frame_num in near_frames:
            if episodes and frame_num - episodes[-1][1] <= self.max_episode_gap:
                episodes[-1][1] = frame_num
            else:
                episodes.append([frame_num, frame_num])
        return [tuple(episode) for episode in episodes]

    def find_team_in_possession(self, holders, player_teams, hoop_frame):
        """Returns the team whose players held the ball most shortly before it reached the hoop, or None."""
        counts = Counter(player_teams[t].get(holders[t])
                         for t in range(max(0, hoop_frame - self.possession_frames), hoop_frame + 1)
                         if holders[t] is not None)
        counts.pop(None, None)
        return counts.most_common(1)[0][0] if counts else None

    def find_release(self, balls, player_tracks, hoop_frame, player_teams=None, team=None):
        """
        Finds the shooter and the last frame they held the ball before (or while) it reached
        the hoop. On a dunk or layup the shooter still holds the ball at the hoop. If a team is
        given, only its players can hold the ball.

        Returns:
            tuple: (release frame, shooter id), or (None, None) if no shooter is found.
        """
        holders = {}

        def holder(frame_num):
            if frame_num not in holders:
                players = player_tracks[frame_num]
                if team is not None:
                    players = {player_id: player for player_id, player in players.items()
                               if player_teams[frame_num].get(player_id) == team}
                holders[frame_num] = self.get_ball_holder(balls[frame_num], players)
            return holders[frame_num]

        first_frame = max(0, hoop_frame - self.max_flight_frames)
        for frame_num in range(hoop_frame, first_frame - 1, -1):
            shooter_id = holder(frame_num)
            if shooter_id is None:
                continue
            start = frame_num - self.min_hold_frames + 1
            if start >= 0 and all(holder(t) == shooter_id for t in range(start, frame_num + 1)):
                return frame_num, shooter_id
        return None, None

    def find_take_off(self, player_tracks, shooter_id, release_frame):
        """
        Estimates the take-off frame as the frame shortly before the release where the shooter's
        feet are lowest in the image, i.e. still on the floor.
        """
        earliest, latest = self.take_off_window
        candidates = []
        for frame_num in range(max(0, release_frame - earliest), max(0, release_frame - latest) + 1):
            player = player_tracks[frame_num].get(shooter_id)
            if player is not None:
                candidates.append((player['bbox'][3], frame_num))
        return max(candidates)[1] if candidates else release_frame

    def get_court_position(self, player_tracks, homographies, player_id, frame_num, search=10):
        """
        Maps a player's feet onto the court (meters), using the nearest frame that has both the
        player and a homography.
        """
        for offset in sorted(range(-search, search + 1), key=abs):
            t = frame_num + offset
            if t < 0 or t >= len(homographies) or homographies[t] is None:
                continue
            player = player_tracks[t].get(player_id)
            if player is None:
                continue
            foot = np.array([[get_foot_position(player['bbox'])]], dtype=np.float32)
            tactical = cv2.perspectiveTransform(foot, homographies[t])[0][0]
            return self.tactical_view_converter.tactical_to_meters(tactical)
        return None

    def get_ball_holder(self, ball, players):
        """
        Returns the id of the player whose (slightly enlarged) box contains the ball, preferring
        the player whose upper body is closest to it, or None.
        """
        if ball is None:
            return None
        best_id, best_distance = None, None
        for player_id, player in players.items():
            x1, y1, x2, y2 = player['bbox']
            margin_x, margin_y = (x2 - x1) * self.hold_margin, (y2 - y1) * self.hold_margin
            if not (x1 - margin_x <= ball[0] <= x2 + margin_x and y1 - margin_y <= ball[1] <= y2 + margin_y):
                continue
            upper_body = ((x1 + x2) / 2, y1 + (y2 - y1) / 3)
            distance = np.hypot(ball[0] - upper_body[0], ball[1] - upper_body[1])
            if best_distance is None or distance < best_distance:
                best_id, best_distance = player_id, distance
        return best_id

    @staticmethod
    def get_center(detection):
        return None if not detection else get_center_of_bbox(detection['bbox'])

    @staticmethod
    def fill_short_gaps(values, max_gap):
        """
        Fills gaps of None up to max_gap long, linearly between the values on both sides.
        Values are points (x, y) or boxes (x1, y1, x2, y2).
        """
        values = list(values)
        known = [i for i, value in enumerate(values) if value is not None]
        for previous, following in zip(known, known[1:]):
            gap = following - previous - 1
            if 0 < gap <= max_gap:
                start, end = np.array(values[previous], dtype=float), np.array(values[following], dtype=float)
                for i in range(previous + 1, following):
                    values[i] = tuple(start + (end - start) * (i - previous) / (following - previous))
        return values
