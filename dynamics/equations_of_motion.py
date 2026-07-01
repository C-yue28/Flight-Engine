import numpy as np
from typing import Optional, Callable
from core import StateVector, Vector3, Quaternion, body_to_inertial
from .integrator import Integrator


class EquationsOfMotion:

    """
    Calculation of state derivatives and forces/moments
    """
    
    def __init__(
        self,
        mass: float = 1000.0,
        inertia: Optional[np.ndarray] = np.eye(3) * 1000.0,
    ):
        self.mass = mass
        self.inertia = inertia
        self.integrator = Integrator()
        
        self._aerodynamic_model = None
        self._propulsion_system = None
        self._gravity_model = None
        self._wind_model = None
        
        self._inertia_rate = np.zeros((3, 3), dtype=np.float64)
        self._mass_rate = 0.0
    
    def set_aerodynamic_model(self, model):
        self._aerodynamic_model = model
    
    def set_propulsion_system(self, system):
        self._propulsion_system = system
    
    def set_gravity_model(self, model):
        self._gravity_model = model
    
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
        
        altitude = -position.z
        density = kwargs.get('density', 1.225)
        control_deflections = kwargs.get('control_deflections', {})
        
        F_aero, M_aero = self._compute_aerodynamic_forces(
            state, **kwargs
        )
        F_prop, M_prop = self._compute_propulsion_forces(state)
        F_gravity = self._compute_gravity_force(altitude, attitude) if density > 0 else np.zeros(3)
        
        F_total = F_aero + F_prop + F_gravity
        M_total = M_aero + M_prop
        
        coriolis = angular_velocity.cross(velocity)
        F_total_vector = Vector3.from_array(F_total)
        dv_dt = (F_total_vector / self.mass) - coriolis
        
        I_omega = self.inertia @ angular_velocity.to_array()
        gyroscopic_moment = angular_velocity.cross(Vector3.from_array(I_omega))
        inertia_change_moment = self._inertia_rate @ angular_velocity.to_array()
        
        M_effective = M_total - gyroscopic_moment.to_array() - inertia_change_moment
        domega_dt = np.linalg.solve(self.inertia, M_effective)
        
        R = attitude.to_rotation_matrix()
        dp_dt = R.T @ velocity.to_array()
        
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
                state.position, state.time, -state.position.z
            )
            wind_body = state.attitude.rotate_vector(wind_velocity)
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
        
        mach = kwargs.get('mach', 0.0)
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
        if self._propulsion_system is None:
            return np.zeros(3), np.zeros(3)
        
        return self._propulsion_system.get_total_forces_and_moments(
            state.angular_velocity
        )
    
    def _compute_gravity_force(self, altitude: float, attitude: Quaternion) -> np.ndarray:
        if self._gravity_model is None:
            g_inertial = Vector3(0.0, 0.0, 9.80665)
            g_body = attitude.rotate_vector(g_inertial)
            return g_body.to_array() * self.mass
            
        if self._gravity_model is None:
            g_inertial = Vector3(0.0, 0.0, 9.80665)
            g_body = attitude.rotate_vector(g_inertial)
            return g_body.to_array() * self.mass
        
        g_body = self._gravity_model.vector_body(altitude, attitude)
        return g_body * self.mass
    
    def integrate(
        self,
        state: StateVector,
        dt: float,
        **kwargs
    ) -> StateVector:
        if self._propulsion_system is not None:
            self._propulsion_system.update(dt)
        
        new_state = self.integrator.integrate(
            state, self.derivatives, dt, **kwargs
        )
        
        if self._mass_rate != 0.0:
            new_state.mass = max(0.0, state.mass - self._mass_rate * dt)
            new_state.inertia = self._calculate_inertia(new_state.mass)
        
        return new_state
