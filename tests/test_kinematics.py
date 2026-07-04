import numpy as np
import pytest
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from core import StateVector, Quaternion, Vector3, quaternion_rate, body_to_inertial, inertial_to_body
from dynamics import EquationsOfMotion

""" I was not very confident of the quaternion implementation as I used AI to implement it, so I took some time to write
this kinematics test module """

class TestQuaternionOrthogonality:
    
    """ Quaternion normal should remain near 1 over a long period of time """
    def test_quaternion_orthogonality_long_integration(self):
        state = StateVector(
            position=Vector3(0.0, 0.0, -1000.0),
            velocity=Vector3(100.0, 10.0, 5.0),
            attitude=Quaternion.from_euler(0.1, 0.2, 0.3),
            angular_velocity=Vector3(0.5, 0.3, 0.7), 
            mass=1000.0,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        eom = EquationsOfMotion(
            mass=state.mass,
            inertia=state.inertia,
        )
        
        dt = 0.01
        steps = 10000
        
        for i in range(steps):            
            q_rate = quaternion_rate(state.attitude, state.angular_velocity)
            state.attitude.w += q_rate.w * dt
            state.attitude.x += q_rate.x * dt
            state.attitude.y += q_rate.y * dt
            state.attitude.z += q_rate.z * dt
            
            state.attitude = state.attitude.normalize()
            
            if i % 1000 == 0:
                norm = np.sqrt(
                    state.attitude.w**2 + state.attitude.x**2 + 
                    state.attitude.y**2 + state.attitude.z**2
                )
                deviation = abs(norm - 1.0)
                assert deviation < 1e-6
        
        final_norm = np.sqrt(
            state.attitude.w**2 + state.attitude.x**2 + 
            state.attitude.y**2 + state.attitude.z**2
        )
        final_deviation = abs(final_norm - 1.0)
        assert final_deviation < 1e-7
    
    def test_quaternion_norm_after_rotations(self):
        q = Quaternion.identity()
        
        for i in range(100):
            axis = Vector3(np.random.randn(), np.random.randn(), np.random.randn()).normalize()
            angle = np.random.uniform(-np.pi, np.pi)
            q_rot = Quaternion.from_axis_angle(axis, angle)
            q = q_rot * q
            q = q.normalize()
        
        norm = np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        assert abs(norm - 1.0) < 1e-10

""" Test all of the coordinate transforms in case I flipped a convention again """

class TestCoordinateTransformations:
    
    def test_body_and_inertial(self):
        v_body = Vector3(10.0, 5.0, -3.0) # random vector
        
        attitude = Quaternion.from_euler(0.3, -0.2, 0.5)
        
        v_inertial = body_to_inertial(v_body, attitude)
        v_body_back = inertial_to_body(v_inertial, attitude)
        
        assert np.isclose(v_body.x, v_body_back.x, atol=1e-10)
        assert np.isclose(v_body.y, v_body_back.y, atol=1e-10)
        assert np.isclose(v_body.z, v_body_back.z, atol=1e-10)
    
    def test_quaternion_rotation_consistency(self): 
        q = Quaternion.from_euler(0.5, -0.3, 0.7)
        
        v = Vector3(3.0, 4.0, 5.0)
        
        v_q = q.rotate_vector(v)
        
        R = q.to_rotation_matrix()
        v_R = Vector3.from_array(R @ v.to_array())
        
        # Rotation using either matrix or quaternion should not matter and they should be equal
        assert np.isclose(v_q.x, v_R.x, atol=1e-10)
        assert np.isclose(v_q.y, v_R.y, atol=1e-10)
        assert np.isclose(v_q.z, v_R.z, atol=1e-10)
