"""
Numerical integrators for equations of motion
Supports only Runge-Kutta 4th order integration method for now, I didn't see a point in adding other methods
"""

from typing import Callable
import logging
from core import StateVector

logger = logging.getLogger("flight_engine.dynamics")

class Integrator:

    
    def integrate(
        self,
        state: StateVector,
        derivatives_func: Callable,
        dt: float,
        **kwargs
    ) -> StateVector:
        return self._rk4(state, derivatives_func, dt, **kwargs)

    """
    Runge-Kutta 4th order integration method
    I kind of understand what I built here, but 
    I mainly relied on the source below for the integration method:
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    I want to say this is an elegant and beautiful solution I've built where I can integrate all state
    variables simultaneously, but it's probably a common thing in physics engines.
    """
    
    def _rk4(
        self,
        state: StateVector,
        derivatives_func: Callable,
        dt: float,
        **kwargs
    ) -> StateVector:
        y = state.to_flat_array()    
        k1 = derivatives_func(state, **kwargs)
        
        state_k2 = StateVector.from_flat_array(y + dt * k1 / 2, state.mass, state.inertia, state.time)
        state_k2.normalize_quaternion()
        k2 = derivatives_func(state_k2, **kwargs)
        
        state_k3 = StateVector.from_flat_array(y + dt * k2 / 2, state.mass, state.inertia, state.time)
        state_k3.normalize_quaternion()
        k3 = derivatives_func(state_k3, **kwargs)
        
        state_k4 = StateVector.from_flat_array(y + dt * k3, state.mass, state.inertia, state.time)
        state_k4.normalize_quaternion()
        k4 = derivatives_func(state_k4, **kwargs)
        
        y_new = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        new_state = StateVector.from_flat_array(
            y_new, state.mass, state.inertia, state.time + dt
        )
        
        # Normalize quaternion and check for issues
        old_norm = (new_state.attitude.w**2 + new_state.attitude.x**2 + 
                   new_state.attitude.y**2 + new_state.attitude.z**2)
        new_state.normalize_quaternion()
        new_norm = (new_state.attitude.w**2 + new_state.attitude.x**2 + 
                   new_state.attitude.y**2 + new_state.attitude.z**2)
        
        if old_norm > 1e6 or old_norm < 1e-6:
            logger.warning(f"Quaternion norm anomaly: old={old_norm:.2e}, new={new_norm:.6f}")
        
        # Ensure quaternion w is positive to avoid sign ambiguity
        if new_state.attitude.w < 0:
            new_state.attitude.w = -new_state.attitude.w
            new_state.attitude.x = -new_state.attitude.x
            new_state.attitude.y = -new_state.attitude.y
            new_state.attitude.z = -new_state.attitude.z
        
        return new_state