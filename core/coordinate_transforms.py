"""
Coordinate system transformations for flight simulation
inertial, body, and wind frames
"""

import numpy as np
from typing import Tuple
from .vector3 import Vector3
from .quaternion import Quaternion


def inertial_to_body(vector: Vector3, attitude: Quaternion) -> Vector3:
    return attitude.rotate_vector(vector)

def body_to_inertial(vector: Vector3, attitude: Quaternion) -> Vector3:
    q_inv = attitude.conjugate()
    return q_inv.rotate_vector(vector)


def compute_alpha_beta(velocity_body: Vector3) -> Tuple[float, float]:
    u, v, w = velocity_body.x, velocity_body.y, velocity_body.z
    v_mag = velocity_body.magnitude()
    
    if v_mag < 1e-12:
        return 0.0, 0.0
    
    # attack angle
    alpha = np.arctan2(w, u)
    
    # sideslip angle
    beta = np.arcsin(np.clip(v / v_mag, -1.0, 1.0))
    
    return alpha, beta

"""
Found these wind/body transformations online
"""

def wind_to_body(vector_wind: Vector3, alpha: float, beta: float) -> Vector3:
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    
    R_wind_to_body = np.array([
        [ca * cb, -ca * sb, -sa],
        [sb, cb, 0],
        [sa * cb, -sa * sb, ca]
    ], dtype=np.float64)
    
    v_arr = vector_wind.to_array()
    v_body_arr = R_wind_to_body @ v_arr
    
    return Vector3.from_array(v_body_arr)


def body_to_wind(vector_body: Vector3, alpha: float, beta: float) -> Vector3:
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    
    R_body_to_wind = np.array([
        [ca * cb, sb, sa * cb],
        [-ca * sb, cb, -sa * sb],
        [-sa, 0, ca]
    ], dtype=np.float64)
    
    v_arr = vector_body.to_array()
    v_wind_arr = R_body_to_wind @ v_arr
    
    return Vector3.from_array(v_wind_arr)


def body_velocity_from_airspeed(airspeed: float, alpha: float, beta: float = 0.0) -> Vector3:
    u = airspeed * np.cos(alpha) * np.cos(beta)
    v = airspeed * np.sin(beta)
    w = airspeed * np.sin(alpha) * np.cos(beta)
    
    return Vector3(u, v, w)


def normalize_angle(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))

"""
Some evil looking quaternion kinematics that I used AI to implement
"""

def quaternion_rate(quaternion: Quaternion, angular_velocity: Vector3) -> Quaternion:

    p, q, r = angular_velocity.x, angular_velocity.y, angular_velocity.z
    
    # Skew-symmetric matrix for quaternion kinematics
    # Ω = [[0, -p, -q, -r],
    #      [p,  0,  r, -q],
    #      [q, -r,  0,  p],
    #      [r,  q, -p,  0]]
    
    w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z
    
    qw_dot = 0.5 * (-p * x - q * y - r * z)
    qx_dot = 0.5 * (p * w + r * y - q * z)
    qy_dot = 0.5 * (q * w - r * x + p * z)
    qz_dot = 0.5 * (r * w + q * x - p * y)
    
    return Quaternion(qw_dot, qx_dot, qy_dot, qz_dot)


def angular_velocity_from_quaternion_rate(quaternion: Quaternion, quaternion_rate: Quaternion) -> Vector3:
    q_conj = quaternion.conjugate()
    omega_quat = q_conj * quaternion_rate
    omega_quat = Quaternion(
        2 * omega_quat.w,
        2 * omega_quat.x,
        2 * omega_quat.y,
        2 * omega_quat.z
    )
    
    return Vector3(omega_quat.x, omega_quat.y, omega_quat.z)
