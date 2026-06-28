"""
Core data structures and mathematical utilities for 6DOF flight simulation.
"""

from .state_vector import StateVector
from .vector3 import Vector3
from .matrix3 import Matrix3
from .quaternion import Quaternion
from .coordinate_transforms import (
    inertial_to_body, body_to_inertial,
    compute_alpha_beta, wind_to_body, body_to_wind,
    body_velocity_from_airspeed, normalize_angle,
    quaternion_rate, angular_velocity_from_quaternion_rate
)

__all__ = [
    'StateVector', 'Vector3', 'Matrix3', 'Quaternion',
    'inertial_to_body', 'body_to_inertial',
    'compute_alpha_beta', 'wind_to_body', 'body_to_wind',
    'body_velocity_from_airspeed', 'normalize_angle',
    'quaternion_rate', 'angular_velocity_from_quaternion_rate'
]
