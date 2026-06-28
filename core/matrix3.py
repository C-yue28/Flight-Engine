"""
3x3 Matrix class for inertia tensors and rotation matrices
"""

import numpy as np
from typing import Union, Optional


class Matrix3:    
    def __init__(self, data: Optional[np.ndarray] = None):
        if data is None:
            self.data = np.eye(3, dtype=np.float64)
        else:
            self.data = np.array(data, dtype=np.float64).reshape(3, 3)
    
    @classmethod
    def identity(cls) -> 'Matrix3':
        return cls(np.eye(3, dtype=np.float64))
    
    @classmethod
    def zeros(cls) -> 'Matrix3':
        return cls(np.zeros((3, 3), dtype=np.float64))
    
    @classmethod
    def from_diagonal(cls, d0: float, d1: float, d2: float) -> 'Matrix3':
        return cls(np.diag([d0, d1, d2]))
    
    def copy(self) -> 'Matrix3':
        return Matrix3(self.data.copy())
    
    def inverse(self) -> 'Matrix3':
        try:
            return Matrix3(np.linalg.inv(self.data))
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix")
    
    def transpose(self) -> 'Matrix3':
        return Matrix3(self.data.T)
    
    def determinant(self) -> float:
        return float(np.linalg.det(self.data))
    
    def trace(self) -> float:
        return float(np.trace(self.data))
    
    def __mul__(self, other) -> 'Matrix3':
        if isinstance(other, Matrix3):
            return Matrix3(self.data @ other.data)
        else:
            return Matrix3(self.data * other)

    def __add__(self, other: 'Matrix3') -> 'Matrix3':
        return Matrix3(self.data + other.data)
    
    def __sub__(self, other: 'Matrix3') -> 'Matrix3':
        return Matrix3(self.data - other.data)
    
    def __getitem__(self, key) -> float:
        return self.data[key]
    
    def __setitem__(self, key, value: float):
        self.data[key] = value
    
    def __repr__(self) -> str:
        return f"Matrix3(\n{self.data}\n)"
