"""
Complete state vector: position, velocity, attitude, angular velocity, etc
"""

import numpy as np
from typing import Optional
from .vector3 import Vector3
from .quaternion import Quaternion


class StateVector:
    
    __slots__ = [
        'position',
        'velocity', 
        'attitude',  
        'angular_velocity',  
        'mass',          
        'inertia',       
        'time'           
    ]
    
    def __init__(
        self,
        position: Optional[Vector3] = None,
        velocity: Optional[Vector3] = None,
        attitude: Optional[Quaternion] = None,
        angular_velocity: Optional[Vector3] = None,
        mass: float = 1000.0,
        inertia: Optional[np.ndarray] = None,
        time: float = 0.0
    ):
        self.position = position.copy() if position is not None else Vector3.zeros()
        self.velocity = velocity.copy() if velocity is not None else Vector3.zeros()
        self.attitude = attitude.copy() if attitude is not None else Quaternion.identity()
        self.angular_velocity = angular_velocity.copy() if angular_velocity is not None else Vector3.zeros()
        self.mass = float(mass)
        
        if inertia is None:
            self.inertia = np.eye(3, dtype=np.float64) * 1000.0
        else:
            self.inertia = np.array(inertia, dtype=np.float64).reshape(3, 3)
        
        self.time = float(time)
    
    @classmethod
    def from_flat_array(cls, arr: np.ndarray, mass: float = 1000.0, 
                       inertia: Optional[np.ndarray] = None, time: float = 0.0) -> 'StateVector':
        if len(arr) != 13:
            raise ValueError("Must be 13 elements")
        
        return cls(
            position=Vector3(arr[0], arr[1], arr[2]),
            velocity=Vector3(arr[3], arr[4], arr[5]),
            attitude=Quaternion(arr[6], arr[7], arr[8], arr[9]),
            angular_velocity=Vector3(arr[10], arr[11], arr[12]),
            mass=mass,
            inertia=inertia,
            time=time
        )
    
    def to_flat_array(self) -> np.ndarray:
        return np.array([
            self.position.x, self.position.y, self.position.z,
            self.velocity.x, self.velocity.y, self.velocity.z,
            self.attitude.w, self.attitude.x, self.attitude.y, self.attitude.z,
            self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z
        ], dtype=np.float64)
    
    def copy(self) -> 'StateVector':
        return StateVector(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            attitude=Quaternion(self.attitude.w, self.attitude.x, 
                             self.attitude.y, self.attitude.z),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass,
            inertia=self.inertia.copy(),
            time=self.time
        )
    
    def normalize_quaternion(self) -> None:
        self.attitude = self.attitude.normalize()
    
    def get_altitude(self) -> float:
        return self.position.z
    
    def get_airspeed(self) -> float:
        return self.velocity.magnitude()
    
    def get_rotation_matrix(self) -> np.ndarray:
        return self.attitude.to_rotation_matrix()
    
    def get_euler_angles(self) -> tuple:
        return self.attitude.to_euler()
    
    def rotate_to_inertial(self, body_vector: Vector3) -> Vector3:
        return self.attitude.rotate_vector(body_vector)
    
    def rotate_to_body(self, inertial_vector: Vector3) -> Vector3:
        q_inv = self.attitude.conjugate()
        return q_inv.rotate_vector(inertial_vector)
    
    def __repr__(self) -> str:
        return (f"StateVector(pos={self.position}, vel={self.velocity}, "
                f"att={self.attitude}, ang_vel={self.angular_velocity}, "
                f"mass={self.mass:.2f}, time={self.time:.3f})")
