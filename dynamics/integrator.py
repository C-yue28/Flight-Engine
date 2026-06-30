"""
Numerical integrators for equations of motion
Supports only Runge-Kutta 4th order integration method for now, I didn't see a point in adding other methods
"""

import numpy as np
from typing import Callable
from core import StateVector

class Integrator:

    
    def integrate(
        self,
        state: StateVector,
        derivatives_func: Callable,
        dt: float,
        **kwargs
    ) -> StateVector:
        return self._rk4(state, derivatives_func, dt, **kwargs)
    
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
        k2 = derivatives_func(state_k2, **kwargs)
        
        state_k3 = StateVector.from_flat_array(y + dt * k2 / 2, state.mass, state.inertia, state.time)
        k3 = derivatives_func(state_k3, **kwargs)
        
        state_k4 = StateVector.from_flat_array(y + dt * k3, state.mass, state.inertia, state.time)
        k4 = derivatives_func(state_k4, **kwargs)
        
        y_new = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        new_state = StateVector.from_flat_array(
            y_new, state.mass, state.inertia, state.time + dt
        )
        new_state.normalize_quaternion()
        
        return new_state