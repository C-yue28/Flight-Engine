import numpy as np
import logging
from typing import Optional, Callable
from core import StateVector, Vector3, Quaternion, body_to_inertial, inertial_to_body
from .integrator import Integrator

logger = logging.getLogger("flight_engine.dynamics")


class EquationsOfMotion:

    """
    Calculation of state derivatives and forces/moments
    """
    
    def __init__(
        self,
        mass: float = 1000.0,
        inertia: Optional[np.ndarray] = None
    ):
        self.mass = mass
        self.inertia = inertia if inertia is not None else np.eye(3)
        self.integrator = Integrator()
        
        self._aerodynamic_model = None
        self._propulsion_system = None
        self._gravity_model = None
        self._wind_model = None
        
        self._inertia_rate = np.zeros((3, 3), dtype=np.float64)
        self._mass_rate = 0.0

        self._gravity_enabled = True

    def estimate_inertia_tensor(self) -> np.ndarray:
        if self._aerodynamic_model is None:
            return np.eye(3)

        Lx = self._aerodynamic_model.mean_aerodynamic_chord
        Ly = self._aerodynamic_model.reference_span

        ixx = (1.0 / 12.0) * self.mass * (Ly ** 2)
        iyy = (1.0 / 12.0) * self.mass * (Lx ** 2)
        izz = (1.0 / 12.0) * self.mass * (Lx ** 2 + Ly ** 2)

        self.inertia = np.diag([ixx, iyy, izz]).astype(np.float64)

        return self.inertia

    def set_aerodynamic_model(self, model):
        self._aerodynamic_model = model
    
    def set_propulsion_system(self, system):
        self._propulsion_system = system
    
    def set_gravity_model(self, model):
        self._gravity_model = model

    def turn_gravity_off(self) -> None:
        self._gravity_model = None
        self._gravity_enabled = False
    
    def set_wind_model(self, model):
        self._wind_model = model
    
    def set_mass_properties(self, mass: float, inertia: np.ndarray, 
                          mass_rate: float = 0.0, inertia_rate: Optional[np.ndarray] = None):
        self.mass = mass
        self.inertia = np.array(inertia, dtype=np.float64).reshape(3, 3)
        self._mass_rate = mass_rate
        self._inertia_rate = inertia_rate if inertia_rate is not None else np.zeros((3, 3))
    
    def derivatives(self, state: StateVector, **kwargs) -> np.ndarray:
        position = state.position
        velocity = state.velocity
        attitude = state.attitude
        angular_velocity = state.angular_velocity
        
        altitude = position.z
        density = kwargs.get('density', 1.225)
        control_deflections = kwargs.get('control_deflections', {})
        
        # Check for invalid state
        if np.isnan(position.to_array()).any() or np.isnan(velocity.to_array()).any():
            logger.error("NaN detected in state derivatives")
            return np.zeros(13)
        
        F_aero, M_aero = self._compute_aerodynamic_forces(
            state, **kwargs
        )
        F_prop, M_prop = self._compute_propulsion_forces(state)
        F_gravity = self._compute_gravity_force(altitude, attitude)
        
        F_total = F_aero + F_prop + F_gravity
        M_total = M_aero + M_prop
        
        # Check for force/moment overflow
        if np.isnan(F_total).any() or np.isnan(M_total).any():
            logger.error("NaN in forces/moments")
            return np.zeros(13)
        
        if np.abs(F_total).max() > 1e8:
            logger.warning(f"Excessive force magnitude: {np.abs(F_total).max()}")
        
        coriolis = angular_velocity.cross(velocity)
        F_total_vector = Vector3.from_array(F_total)
        dv_dt = (F_total_vector / self.mass) - coriolis
        
        I_omega = self.inertia @ angular_velocity.to_array()
        gyroscopic_moment = angular_velocity.cross(Vector3.from_array(I_omega))
        inertia_change_moment = self._inertia_rate @ angular_velocity.to_array()
        
        M_effective = M_total - gyroscopic_moment.to_array() - inertia_change_moment
        
        # Check inertia matrix condition
        try:
            domega_dt = np.linalg.solve(self.inertia, M_effective)
        except np.linalg.LinAlgError:
            logger.error("Singular inertia matrix in derivatives")
            return np.zeros(13)
        
        R = attitude.to_rotation_matrix()
        dp_dt = R @ velocity.to_array()
        
        from core import quaternion_rate
        dq_dt = quaternion_rate(attitude, angular_velocity)
        
        deriv = np.array([
            dp_dt[0], dp_dt[1], dp_dt[2],      # Position derivatives
            dv_dt.x, dv_dt.y, dv_dt.z,         # Velocity derivatives
            dq_dt.w, dq_dt.x, dq_dt.y, dq_dt.z,  # Quaternion derivatives
            domega_dt[0], domega_dt[1], domega_dt[2]  # Angular velocity derivatives
        ], dtype=np.float64)
        
        return deriv
    
    def _compute_aerodynamic_forces(
        self,
        state: StateVector,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._aerodynamic_model is None:
            return np.zeros(3), np.zeros(3)
        
        control_deflections = kwargs.get('control_deflections', {})
        density = kwargs.get('density', 1.225)
        
        if self._wind_model is not None:
            wind_velocity = self._wind_model.get_wind_velocity(
                state.position, state.time, state.position.z
            )
            wind_body = inertial_to_body(wind_velocity, state.attitude)
            air_velocity = state.velocity - wind_body
        else:
            air_velocity = state.velocity
        
        air_state = StateVector(
            position=state.position,
            velocity=air_velocity,
            attitude=state.attitude,
            angular_velocity=state.angular_velocity,
            mass=state.mass,
            inertia=state.inertia,
            time=state.time
        )
        
        mach = kwargs.get('mach', air_velocity.magnitude() / 343.0)
        reynolds = kwargs.get('reynolds', 1e6)
        
        return self._aerodynamic_model.compute_forces_and_moments(
            air_state, control_deflections, density, mach, reynolds
        )
    
    def _compute_propulsion_forces(self, state: StateVector) -> tuple[np.ndarray, np.ndarray]:
        if self._propulsion_system is None:
            return np.zeros(3), np.zeros(3)
        
        return self._propulsion_system.get_total_forces_and_moments(
            state.angular_velocity
        )
    
    def _compute_gravity_force(self, altitude: float, attitude: Quaternion) -> np.ndarray:
        g_body = np.zeros(3)

        if not self._gravity_enabled:
            return g_body

        if self._gravity_model is None:
            g_inertial = Vector3(0.0, 0.0, 9.80665)
            g_body = inertial_to_body(g_inertial, attitude)
            return g_body.to_array() * self.mass
        else:
            g_body = self._gravity_model.vector_body(altitude, attitude)
        return g_body * self.mass

    def integrate(
        self,
        state: StateVector,
        dt: float,
        **kwargs
    ) -> StateVector:
        
        new_state = self.integrator.integrate(
            state, self.derivatives, dt, **kwargs
        )
        
        if self._mass_rate != 0.0:
            print("-------------------------------------\nTESTING\n----------------------------------------")
            new_state.mass = max(0.0, state.mass - self._mass_rate * dt)
            self.mass = new_state.mass
            new_state.inertia = self.estimate_inertia_tensor()
        
        return new_state
