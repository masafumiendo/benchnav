"""
Masafumi Endo, 2024.
"""

from __future__ import annotations
from typing import Union

import numpy as np
import torch


class Dubins:
    """
    A class to represent the Dubins pathfinding algorithm.
    """

    def __init__(self, radius: float, resolution: float) -> None:
        """
        Initialize the Dubins pathfinding algorithm.

        Parameters:
        - radius (float): Radius of the circles for the turns.
        - resolution (float): Resolution of the consecutive points in the path.
        """
        assert radius > 0 and resolution > 0  # Check for positive values
        self._radius = radius
        self._resolution = resolution

    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute the Euclidean distance between two points.

        Parameters:
        - p1 (np.ndarray): First point.
        - p2 (np.ndarray): Second point.

        Returns:
        - distance (float): Euclidean distance between the two points.
        """
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def _orthogonal(vect2d: np.ndarray) -> np.ndarray:
        """
        Compute an orthogonal vector to the one given.

        Parameters:
        - vect2d (np.ndarray): 2D vector.

        Returns:
        - orthogonal (np.ndarray): Orthogonal 2D vector.
        """
        return np.array((-vect2d[1], vect2d[0]))

    def dubins_path(
        self,
        start: Union[np.ndarray, torch.Tensor],
        end: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute the shortest Dubins path between two points.

        Parameters:
        - start (Union[np.ndarray, torch.Tensor]): Initial point.
        - end (Union[np.ndarray, torch.Tensor]): Final point.

        Returns:
        - Union[np.ndarray, torch.Tensor]: Succession of points representing the shortest Dubins path.
        """
        device = start.device if isinstance(start, torch.Tensor) else None
        start = start.cpu().numpy() if isinstance(start, torch.Tensor) else start
        end = end.cpu().numpy() if isinstance(end, torch.Tensor) else end
        options = self.all_options(start, end)
        shortest = min(options, key=lambda x: x[0])
        points = self.generate_points(start, end, shortest[1], shortest[2])
        return torch.tensor(points, device=device) if device else points

    def all_options(
        self, start: np.ndarray, end: np.ndarray, sort: bool = False
    ) -> list[tuple[float, np.ndarray, bool]]:
        """
        Computes all the possible Dubin's path and returns the sequence of
        points representing the shortest option.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - sort (bool): Whether to sort the options by distance.

        Returns:
        - list[tuple[float, np.ndarray, bool]]: List of tuples containing the total distance of the path,
        the path itself, and a boolean indicating if the path is feasible.
        """
        centers = {
            "left_start": self.find_center(start, "L"),
            "right_start": self.find_center(start, "R"),
            "left_end": self.find_center(end, "L"),
            "right_end": self.find_center(end, "R"),
        }

        options = [
            self.lsl(start, end, centers["left_start"], centers["left_end"]),
            self.rsr(start, end, centers["right_start"], centers["right_end"]),
            self.rsl(start, end, centers["right_start"], centers["left_end"]),
            self.lsr(start, end, centers["left_start"], centers["right_end"]),
            self.rlr(start, end, centers["right_start"], centers["right_end"]),
            self.lrl(start, end, centers["left_start"], centers["left_end"]),
        ]

        if sort:
            options.sort(key=lambda x: x[0])

        return options

    def generate_points(
        self,
        start: np.ndarray,
        end: np.ndarray,
        dubins_path: np.ndarray,
        straight: bool,
    ) -> np.ndarray:
        """
        Transforms the Dubins path in a succession of points in the 2D plane.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - dubins_path (np.ndarray): The Dubins path in the form of a tuple containing the angles of the turns.
        - straight (bool): Whether there is a straight segment in the path.

        Returns:
        - np.ndarray: Succession of points representing the shortest Dubins path.
        """
        if straight:
            return self.generate_points_straight(start, end, dubins_path)
        return self.generate_points_curve(start, end, dubins_path)

    def lsl(
        self,
        start: np.ndarray,
        end: np.ndarray,
        center_0: np.ndarray,
        center_2: np.ndarray,
    ) -> tuple[float, np.ndarray, bool]:
        """
        Left-Straight-Left trajectories.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - center_0 (np.ndarray): Center of the first turn.
        - center_2 (np.ndarray): Center of the last turn.

        Returns:
        - tuple[float, np.ndarray, bool]: Tuple containing the total distance of the path,
        """
        straight_dist = self._distance(center_0, center_2)
        alpha = np.arctan2(center_2[1] - center_0[1], center_2[0] - center_0[0])
        beta_2 = (end[2] - alpha) % (2 * np.pi)
        beta_0 = (alpha - start[2]) % (2 * np.pi)
        total_len = self._radius * (beta_2 + beta_0) + straight_dist
        return total_len, np.array([beta_0, beta_2, straight_dist]), True

    def rsr(
        self,
        start: np.ndarray,
        end: np.ndarray,
        center_0: np.ndarray,
        center_2: np.ndarray,
    ) -> tuple[float, np.ndarray, bool]:
        """
        Right-Straight-Right trajectories.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - center_0 (np.ndarray): Center of the first turn.
        - center_2 (np.ndarray): Center of the last turn.

        Returns:
        - tuple[float, np.ndarray, bool]: Tuple containing the total distance of the path,
        """
        straight_dist = self._distance(center_0, center_2)
        alpha = np.arctan2(center_2[1] - center_0[1], center_2[0] - center_0[0])
        beta_2 = (-end[2] + alpha) % (2 * np.pi)
        beta_0 = (-alpha + start[2]) % (2 * np.pi)
        total_len = self._radius * (beta_2 + beta_0) + straight_dist
        return total_len, np.array([-beta_0, -beta_2, straight_dist]), True

    def rsl(
        self,
        start: np.ndarray,
        end: np.ndarray,
        center_0: np.ndarray,
        center_2: np.ndarray,
    ) -> tuple[float, np.ndarray, bool]:
        """
        Right-Straight-Left trajectories.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - center_0 (np.ndarray): Center of the first turn.
        - center_2 (np.ndarray): Center of the last turn.

        Returns:
        - tuple[float, np.ndarray, bool]: Tuple containing the total distance of the path,
        """
        median_point = (center_2 - center_0) / 2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self._radius:
            return (float("inf"), np.zeros(3), True)
        alpha = np.arccos(self._radius / half_intercenter)
        beta_0 = -(psia + alpha - start[2] - np.pi / 2) % (2 * np.pi)
        beta_2 = (np.pi + end[2] - np.pi / 2 - alpha - psia) % (2 * np.pi)
        straight_dist = 2 * np.sqrt(half_intercenter**2 - self._radius**2)
        total_len = self._radius * (beta_0 + beta_2) + straight_dist
        return total_len, np.array([-beta_0, beta_2, straight_dist]), True

    def lsr(
        self,
        start: np.ndarray,
        end: np.ndarray,
        center_0: np.ndarray,
        center_2: np.ndarray,
    ) -> tuple[float, np.ndarray, bool]:
        """
        Left-Straight-Right trajectories.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - center_0 (np.ndarray): Center of the first turn.
        - center_2 (np.ndarray): Center of the last turn.

        Returns:
        - tuple[float, np.ndarray, bool]: Tuple containing the total distance of the path,
        """
        median_point = (center_2 - center_0) / 2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self._radius:
            return (float("inf"), np.zeros(3), True)
        alpha = np.arccos(self._radius / half_intercenter)
        beta_0 = (psia - alpha - start[2] + np.pi / 2) % (2 * np.pi)
        beta_2 = (0.5 * np.pi - end[2] - alpha + psia) % (2 * np.pi)
        straight_dist = 2 * np.sqrt(half_intercenter**2 - self._radius**2)
        total_len = self._radius * (beta_0 + beta_2) + straight_dist
        return total_len, np.array([beta_0, -beta_2, straight_dist]), True

    def lrl(
        self,
        start: np.ndarray,
        end: np.ndarray,
        center_0: np.ndarray,
        center_2: np.ndarray,
    ) -> tuple[float, np.ndarray, bool]:
        """
        Left-right-Left trajectories.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - center_0 (np.ndarray): Center of the first turn.
        - center_2 (np.ndarray): Center of the last turn.

        Returns:
        - tuple[float, np.ndarray, bool]: Tuple containing the total distance of the path,
        """
        intercenter_dist = self._distance(center_0, center_2)
        if intercenter_dist > 4 * self._radius or intercenter_dist < 2 * self._radius:
            return (float("inf"), np.zeros(3), False)
        gamma = 2 * np.arcsin(intercenter_dist / (4 * self._radius))
        beta_0 = (
            np.arctan2(center_2[1] - center_0[1], center_2[0] - center_0[0])
            - start[2]
            + np.pi / 2
            + (np.pi - gamma) / 2
        ) % (2 * np.pi)
        beta_2 = (
            -np.arctan2(center_2[1] - center_0[1], center_2[0] - center_0[0])
            + end[2]
            + np.pi / 2
            + (np.pi - gamma) / 2
        ) % (2 * np.pi)
        total_len = self._radius * (2 * np.pi - gamma + abs(beta_0) + abs(beta_2))
        return total_len, np.array([beta_0, beta_2, 2 * np.pi - gamma]), False

    def rlr(
        self,
        start: np.ndarray,
        end: np.ndarray,
        center_0: np.ndarray,
        center_2: np.ndarray,
    ) -> tuple[float, np.ndarray, bool]:
        """
        Right-left-right trajectories.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - center_0 (np.ndarray): Center of the first turn.
        - center_2 (np.ndarray): Center of the last turn.

        Returns:
        - tuple[float, np.ndarray, bool]: Tuple containing the total distance of the path,
        """
        intercenter_dist = self._distance(center_0, center_2)
        if intercenter_dist > 4 * self._radius or intercenter_dist < 2 * self._radius:
            return (float("inf"), np.zeros(3), False)
        gamma = 2 * np.arcsin(intercenter_dist / (4 * self._radius))
        beta_0 = (
            -np.arctan2(center_2[1] - center_0[1], center_2[0] - center_0[0])
            + start[2]
            + np.pi / 2
            + (np.pi - gamma) / 2
        ) % (2 * np.pi)
        beta_2 = (
            np.arctan2(center_2[1] - center_0[1], center_2[0] - center_0[0])
            - end[2]
            + np.pi / 2
            + (np.pi - gamma) / 2
        ) % (2 * np.pi)
        total_len = self._radius * (2 * np.pi - gamma + abs(beta_0) + abs(beta_2))
        return total_len, np.array([-beta_0, -beta_2, 2 * np.pi - gamma]), False

    def find_center(self, point: np.ndarray, side: str) -> np.ndarray:
        """
        Find the center of the circle that passes through the point and has a radius of self._radius.

        Parameters:
        - point (np.ndarray): Point in the form of (x, y, psi).
        - side (str): Side of the circle, either "L" or "R".

        Returns:
        - np.ndarray: Center of the circle.
        """
        angle = point[2] + (np.pi / 2 if side == "L" else -np.pi / 2)
        return np.array(
            [
                point[0] + np.cos(angle) * self._radius,
                point[1] + np.sin(angle) * self._radius,
            ]
        )

    def generate_points_straight(
        self, start: np.ndarray, end: np.ndarray, path: np.ndarray
    ) -> np.ndarray:
        """
        For the two first paths, where the trajectory is a succession of 2 turns and a straight segment.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - path (np.ndarray): The computed Dubins path.

        Returns:
        - np.ndarray: Succession of points representing the shortest Dubins path.
        """
        total = self._radius * (abs(path[1]) + abs(path[0])) + path[2]  # Path length
        center_0 = self.find_center(start, "L" if path[0] > 0 else "R")
        center_2 = self.find_center(end, "L" if path[1] > 0 else "R")

        if abs(path[0]) > 0:
            angle = start[2] + (abs(path[0]) - np.pi / 2) * np.sign(path[0])
            ini = center_0 + self._radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            ini = np.array(start[:2])

        if abs(path[1]) > 0:
            angle = end[2] + (-abs(path[1]) - np.pi / 2) * np.sign(path[1])
            fin = center_2 + self._radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            fin = np.array(end[:2])
        dist_straight = self._distance(ini, fin)

        points = []
        for x in np.arange(0, total, self._resolution):
            if x < abs(path[0]) * self._radius:
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1]) * self._radius:
                points.append(self.circle_arc(end, path[1], center_2, x - total))
            else:
                coeff = (x - abs(path[0]) * self._radius) / dist_straight
                points.append(coeff * fin + (1 - coeff) * ini)
        points.append(end[:2])
        return np.array(points)

    def generate_points_curve(
        self, start: np.ndarray, end: np.ndarray, path: np.ndarray
    ) -> np.ndarray:
        """
        For the two last paths, where the trajectory is a succession of 3 turns.

        Parameters:
        - start (np.ndarray): Initial point.
        - end (np.ndarray): Final point.
        - path (np.ndarray): The computed Dubins path.

        Returns:
        - np.ndarray: Succession of points representing the shortest Dubins path.
        """
        total = self._radius * (abs(path[1]) + abs(path[0]) + abs(path[2]))
        center_0 = self.find_center(start, "L" if path[0] > 0 else "R")
        center_2 = self.find_center(end, "L" if path[1] > 0 else "R")
        intercenter = self._distance(center_0, center_2)
        center_1 = (center_0 + center_2) / 2 + np.sign(path[0]) * self._orthogonal(
            (center_2 - center_0) / intercenter
        ) * (4 * self._radius**2 - (intercenter / 2) ** 2) ** 0.5
        psi_0 = np.arctan2((center_1 - center_0)[1], (center_1 - center_0)[0]) - np.pi

        points = []
        for x in np.arange(0, total, self._resolution):
            if x < abs(path[0]) * self._radius:
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1]) * self._radius:
                points.append(self.circle_arc(end, path[1], center_2, x - total))
            else:
                angle = psi_0 - np.sign(path[0]) * (x / self._radius - abs(path[0]))
                vect = np.array([np.cos(angle), np.sin(angle)])
                points.append(center_1 + self._radius * vect)
        points.append(end[:2])
        return np.array(points)

    def circle_arc(
        self, reference: np.ndarray, beta: float, center: np.ndarray, x: float
    ) -> np.ndarray:
        """
        Compute the point in the circle arc.

        Parameters:
        - reference (np.ndarray): Reference point.
        - beta (float): Angle of the turn.
        - center (np.ndarray): Center of the circle.
        - x (float): Distance from the center.

        Returns:
        - np.ndarray: Point in the circle arc.
        """
        angle = reference[2] + ((x / self._radius) - np.pi / 2) * np.sign(beta)
        vect = np.array([np.cos(angle), np.sin(angle)])
        return center + self._radius * vect
