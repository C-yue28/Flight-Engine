import numpy as np
from typing import Optional
from core import StateVector, Vector3

"""
Could add spinning and gyroscopic moments later
"""


class PropulsionSystem:

    """
    Basic propulsion system with max thrust and throttle
    """
    
    def __init__(
        self,
        max_thrust: float = 20000.0,
        thrust_direction: Optional[Vector3] = None,
    ):
        self.max_thrust = max_thrust
        
        if thrust_direction is None:
            self.thrust_direction = Vector3(1.0, 0.0, 0.0)  # Forward
        else:
            self.thrust_direction = thrust_direction.normalize()

        self._throttle_command = 0.0
        self._current_throttle = 0.0
    
    def set_throttle(self, throttle: float) -> None: 
        self._throttle_command = np.clip(throttle, 0.0, 1.0)

    def get_thrust_force(self) -> Vector3:
        thrust_magnitude = self._current_throttle * self.max_thrust
        return self.thrust_direction * thrust_magnitude
    
    def get_thrust_moment(self) -> np.ndarray:
        thrust = self.get_thrust_force()
        moment = self.engine_position.cross(thrust)
        return moment.to_array()
    
    def get_total_forces_and_moments(
        self,
        angular_velocity: Vector3
    ) -> tuple[np.ndarray, np.ndarray]:
        forces = self.get_thrust_force().to_array()
        
        moments = self.get_thrust_moment()
                
        return forces, moments
    
    def reset(self):
        self._throttle_command = 0.0
        self._current_throttle = 0.0
        self._engine_rpm = 0.0
