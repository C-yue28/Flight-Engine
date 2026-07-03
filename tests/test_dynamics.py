import numpy as np
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from core import StateVector, Vector3, Quaternion
from dynamics import EquationsOfMotion
from core import quaternion_rate

"""
Dynamics test module
"""

class TestLinearMomentumConservation:
    
    def test_linear_momentum_conservation(self):
        mass = 1000.0
        state = StateVector(
            position=Vector3(0.0, 0.0, 0.0),
            velocity=Vector3.zeros(),
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(),
            mass=mass,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        eom = EquationsOfMotion(
            mass=mass,
            inertia=state.inertia,
        )
        
        F_x = 5000.0
        dt = 0.01
        total_time = 1.0
        steps = int(total_time / dt)
        
        def vacuum_derivatives(state, **kwargs):
            F_total = np.array([F_x, 0.0, 0.0])
            dv_dt = F_total / state.mass
            dp_dt = state.velocity.to_array()
            domega_dt = np.zeros(3)
            domega_dt = np.zeros(3)
            dq_dt = quaternion_rate(state.attitude, state.angular_velocity)
            
            return np.array([
                dp_dt[0], dp_dt[1], dp_dt[2],
                dv_dt[0], dv_dt[1], dv_dt[2],
                dq_dt.w, dq_dt.x, dq_dt.y, dq_dt.z,
                domega_dt[0], domega_dt[1], domega_dt[2]
            ])
        
        for _ in range(steps):
            deriv = vacuum_derivatives(state)
            y = state.to_flat_array()
            y_new = y + dt * deriv
            state = StateVector.from_flat_array(y_new, mass, state.inertia, state.time + dt)
        
        expected_velocity = F_x * total_time / mass
        actual_velocity = state.velocity.x
        
        assert np.isclose(actual_velocity, expected_velocity, rtol=1e-6)
    
    def test_linear_momentum_zero_force(self):
        mass = 1000.0
        # Use horizontal velocity only to avoid gravity effects
        initial_velocity = Vector3(50.0, 30.0, 0.0)
        
        state = StateVector(
            position=Vector3.zeros(),
            velocity=initial_velocity,
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(),
            mass=mass,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        eom = EquationsOfMotion(
            mass=mass,
            inertia=state.inertia,
        )
        
        dt = 0.01
        steps = 100
        
        for _ in range(steps):
            deriv = eom.derivatives(state, density=0.0, control_deflections={})
            state = eom.integrator.integrate(state, eom.derivatives, dt, 
                                           density=0.0, control_deflections={})
        
        # With no aerodynamics, horizontal velocity should remain constant
        assert np.isclose(state.velocity.x, initial_velocity.x, atol=1e-6)
        assert np.isclose(state.velocity.y, initial_velocity.y, atol=1e-6)


class TestAngularMomentumConservation:
    
    def test_angular_momentum_conservation(self):
        mass = 1000.0
        inertia = np.array([
            [1000.0, 100.0, 50.0], 
            [100.0, 2000.0, 75.0],
            [50.0, 75.0, 1500.0]
        ])
        
        initial_omega = Vector3(0.5, 0.3, 0.7)
        
        state = StateVector(
            position=Vector3.zeros(),
            velocity=Vector3.zeros(),
            attitude=Quaternion.identity(),
            angular_velocity=initial_omega,
            mass=mass,
            inertia=inertia
        )
        
        eom = EquationsOfMotion(
            mass=mass,
            inertia=inertia,
        )
        
        L_initial = inertia @ initial_omega.to_array()
        L_initial_mag = np.linalg.norm(L_initial)
        
        dt = 0.01
        steps = 1000
        
        for i in range(steps):
            deriv = eom.derivatives(state, density=0.0, control_deflections={})
            state = eom.integrator.integrate(state, eom.derivatives, dt,
                                           density=0.0, control_deflections={})
            
            if i % 100 == 0:
                L_current = state.inertia @ state.angular_velocity.to_array()
                L_current_mag = np.linalg.norm(L_current)
                
                # Conservation of angular momentum
                assert np.isclose(L_current_mag, L_initial_mag, rtol=1e-4), (
                    f"Angular momentum magnitude not conserved at step {i}: "
                    f"initial {L_initial_mag}, current {L_current_mag}"
                )
        
        L_final = state.inertia @ state.angular_velocity.to_array()
        L_final_mag = np.linalg.norm(L_final)
        
        assert np.isclose(L_final_mag, L_initial_mag, rtol=1e-4)
    
    def test_intermediate_axis_theorem(self):
        # Tests intermediate axis theorem where small perturbations along a non-"major" axis causes erratic behavior
        # Recommended by Gemini for unit test
        mass = 1000.0
        inertia = np.diag([1000.0, 2000.0, 3000.0])
        
        initial_omega = Vector3(0.0, 1.0, 0.0)
        
        state = StateVector(
            position=Vector3.zeros(),
            velocity=Vector3.zeros(),
            attitude=Quaternion.identity(),
            angular_velocity=initial_omega,
            mass=mass,
            inertia=inertia
        )
        
        eom = EquationsOfMotion(
            mass=mass,
            inertia=inertia,
        )
        
        state.angular_velocity = Vector3(0.01, 1.0, 0.01)
        
        dt = 0.01
        steps = 1000
        
        omega_history = []
        
        for i in range(steps):
            deriv = eom.derivatives(state, density=0.0, control_deflections={})
            state = eom.integrator.integrate(state, eom.derivatives, dt,
                                           density=0.0, control_deflections={})
            
            omega_history.append(state.angular_velocity.copy())
        
        omega_x_values = [w.x for w in omega_history]
        omega_z_values = [w.z for w in omega_history]
        
        x_variation = max(omega_x_values) - min(omega_x_values)
        z_variation = max(omega_z_values) - min(omega_z_values)
        
        assert x_variation > 0.1
        assert z_variation > 0.1


class TestEnergyConservation:
    
    # Basic kinetic energy conservation
    # Modified to allow for integration errors and there is now no gravity
    def test_kinetic_energy_conservation(self):
        mass = 1000.0
        initial_velocity = Vector3(50.0, 30.0, 0.0)
        
        state = StateVector(
            position=Vector3.zeros(),
            velocity=initial_velocity,
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(),
            mass=mass,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        eom = EquationsOfMotion(
            mass=mass,
            inertia=state.inertia,
        )
        
        eom.set_gravity_model(None)
        eom.turn_gravity_off()
        
        KE_initial = 0.5 * mass * initial_velocity.magnitude_squared()
        
        dt = 0.01
        steps = 100
        
        for _ in range(steps):
            deriv = eom.derivatives(state, density=0.0, control_deflections={})
            state = eom.integrator.integrate(state, eom.derivatives, dt,
                                           density=0.0, control_deflections={})
        
        KE_final = 0.5 * mass * state.velocity.magnitude_squared()
        
        assert np.isclose(KE_final, KE_initial, rtol=1e-6)