import numpy as np
import logging
from typing import Union, Tuple

logger = logging.getLogger("flight_engine.core")

class Vector3:
    
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    @classmethod
    def from_array(cls, arr: Union[np.ndarray, list, tuple]) -> 'Vector3':
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    @classmethod
    def zeros(cls) -> 'Vector3':
        return cls(0.0, 0.0, 0.0)
    
    @classmethod
    def ones(cls) -> 'Vector3':
        return cls(1.0, 1.0, 1.0)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def copy(self) -> 'Vector3':
        return Vector3(self.x, self.y, self.z)
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def magnitude_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2
    
    def normalize(self) -> 'Vector3':
        mag = self.magnitude()
        if mag < 1e-12:
            return Vector3.zeros()
        return Vector3(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3':
        if abs(scalar) < 1e-12:
            logger.warning(f"Division by near-zero scalar: {scalar}")
            return Vector3.zeros()
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self) -> 'Vector3':
        return Vector3(-self.x, -self.y, -self.z)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3):
            return False
        return (self.x == other.x and self.y == other.y and self.z == other.z)
    
    def __repr__(self) -> str:
        return f"Vector3({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"
