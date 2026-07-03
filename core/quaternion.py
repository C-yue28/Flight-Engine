"""
Quaternion class
"""

import numpy as np
import logging
from typing import Tuple
from .vector3 import Vector3

logger = logging.getLogger("flight_engine.core")

"""
Apparently my experience with quaternions in Unity wasn't enough for this so I had to do a bunch of research and use AI
"""

class Quaternion:
    
    __slots__ = ['w', 'x', 'y', 'z']
    
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    @classmethod
    def identity(cls) -> 'Quaternion':
        return cls(1.0, 0.0, 0.0, 0.0)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        if len(arr) != 4:
            raise ValueError("Array must have exactly 4 elements")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return cls(w, x, y, z)
    
    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle: float) -> 'Quaternion':
        axis = axis.normalize()
        half_angle = angle * 0.5
        sin_half = np.sin(half_angle)
        
        return cls(
            np.cos(half_angle),
            axis.x * sin_half,
            axis.y * sin_half,
            axis.z * sin_half
        )
    
    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)
    
    def copy(self) -> 'Quaternion':
        return Quaternion(self.w, self.x, self.y, self.z)
    
    def to_euler(self) -> Tuple[float, float, float]:
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        
        if w < 0:
            w, x, y, z = -w, -x, -y, -z
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def to_rotation_matrix(self) -> np.ndarray:
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ], dtype=np.float64)
        
        return R
    
    def normalize(self) -> 'Quaternion':
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq < 1e-12:
            logger.warning("Quaternion norm too small, returning identity")
            return Quaternion.identity()
        
        norm = np.sqrt(norm_sq)
        if norm > 1e6:
            logger.warning(f"Quaternion norm very large: {norm}")
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)
    
    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self) -> 'Quaternion':
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq < 1e-12:
            return Quaternion.identity()
        
        inv_norm = 1.0 / norm_sq
        return Quaternion(self.w * inv_norm, -self.x * inv_norm, -self.y * inv_norm, -self.z * inv_norm)
    
    def rotate_vector(self, v: Vector3) -> Vector3:
        q_v = Quaternion(0.0, v.x, v.y, v.z)
        q_conj = self.conjugate()
        q_rotated = self * q_v * q_conj
        return Vector3(q_rotated.x, q_rotated.y, q_rotated.z)
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return False
        return (self.w == other.w and self.x == other.x and 
                self.y == other.y and self.z == other.z)

    def __repr__(self) -> str:
        return f"Quaternion({self.w:.6f}, {self.x:.6f}, {self.y:.6f}, {self.z:.6f})"
    
    def __str__(self) -> str:
        return f"[{self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f}]"
