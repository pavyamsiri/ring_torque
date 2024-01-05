"""The Quaternion class and related functions."""

from __future__ import annotations

# Standard libraries
from dataclasses import dataclass

# External libraries
import numpy as np
from numpy import typing as npt


@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    @staticmethod
    def from_axis_angle(
        axis: tuple[float, float, float] | npt.NDArray[np.floating], angle: float
    ) -> Quaternion:
        """Creates Quaternion that represents a rotation of the given angle about the given axis.

        Parameters
        ----------
        axis : tuple[float, float, float] | npt.NDArray[np.floating]
            The axis of rotation.
        angle : float
            The angle of rotation.

        Returns
        -------
        Quaternion
            The corresponding Quaternion.
        """
        assert len(axis) == 3
        w = np.cos(angle / 2)
        x = axis[0] * np.sin(angle / 2)
        y = axis[1] * np.sin(angle / 2)
        z = axis[2] * np.sin(angle / 2)
        return Quaternion(w, x, y, z)

    @staticmethod
    def create_vector_quaternion(
        vector: tuple[float, float, float] | npt.NDArray[np.floating]
    ) -> Quaternion:
        assert len(vector) == 3
        return Quaternion(0, vector[0], vector[1], vector[2])

    def __add__(self, other: Quaternion | float) -> Quaternion:
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w + other.w,
                self.x + other.x,
                self.y + other.y,
                self.z + other.z,
            )
        else:
            return Quaternion(
                self.w + other,
                self.x,
                self.y,
                self.z,
            )

    def __radd__(self, other: Quaternion | float) -> Quaternion:
        if isinstance(other, Quaternion):
            return other + self
        else:
            return Quaternion(
                self.w + other,
                self.x,
                self.y,
                self.z,
            )

    def __str__(self) -> str:
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def __sub__(self, other: Quaternion | float) -> Quaternion:
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w - other.w,
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
            )
        else:
            return Quaternion(
                self.w - other,
                self.x,
                self.y,
                self.z,
            )

    def __rsub__(self, other: Quaternion | float) -> Quaternion:
        if isinstance(other, Quaternion):
            return other - self
        else:
            return Quaternion(
                other - self.w,
                -self.x,
                -self.y,
                -self.z,
            )

    def __mul__(self, other: Quaternion | float) -> Quaternion:
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.w
                - self.x * other.x
                - self.y * other.y
                - self.z * other.z,
                self.w * other.x
                + self.x * other.w
                - self.y * other.z
                + self.z * other.y,
                self.w * other.y
                + self.x * other.z
                + self.y * other.w
                - self.z * other.x,
                self.w * other.z
                - self.x * other.y
                + self.y * other.x
                + self.z * other.w,
            )
        else:
            return Quaternion(
                self.w * other,
                self.x * other,
                self.y * other,
                self.z * other,
            )

    def __rmul__(self, other: Quaternion | float) -> Quaternion:
        if isinstance(other, Quaternion):
            return other * self
        else:
            return Quaternion(
                self.w * other,
                self.x * other,
                self.y * other,
                self.z * other,
            )

    def __truediv__(self, other: float) -> Quaternion:
        return Quaternion(
            self.w / other,
            self.x / other,
            self.y / other,
            self.z / other,
        )

    def normalise(self) -> Quaternion:
        norm = self.norm()
        if norm > 0:
            return self / norm
        else:
            return self

    def norm(self) -> float:
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def exponential(self) -> Quaternion:
        norm = self.norm()
        if np.isclose(norm, 0):
            return Quaternion(1, 0, 0, 0)
        else:
            return np.cos(norm) + self / norm * np.sin(norm)
