from dataclasses import dataclass, field
import logging

from sys import path
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
path.append(str(parent_dir))

import numpy as np
from core import StateVector, Vector3, Quaternion, compute_alpha_beta
from dynamics import EquationsOfMotion
from aerodynamics import AerodynamicModel
from propulsion import PropulsionSystem
from utils import setup_simulation_logger

logger = setup_simulation_logger(verbose=True, log_to_file=True, log_to_console=True)

@dataclass
class SimulationConfig:

    initial_altitude: float = 1000.0
    initial_velocity: float = 60.0
    duration: float = 120.0
    dt: float = 0.02
    mass: float = 1200.0
    reference_area: float = 18
    reference_span: float = 12
    mean_aerodynamic_chord: float = 1.5
    max_thrust: float = 2000.0

    def __init__(
        self,
        initial_altitude: float,
        initial_velocity: float,
        duration: float,
        dt: float,
        mass: float,
        reference_area: float,
        reference_span: float,
        mean_aerodynamic_chord: float,
        max_thrust: float
    ):
        self.initial_altitude = initial_altitude
        self.initial_velocity = initial_velocity
        self.duration = duration
        self.dt = dt
        self.mass = mass
        self.reference_area = reference_area
        self.reference_span = reference_span
        self.mean_aerodynamic_chord = mean_aerodynamic_chord
        self.max_thrust = max_thrust


class SimulationData:

    time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    position: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    altitude: np.ndarray = field(default_factory=lambda: np.zeros(0))
    airspeed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    roll: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pitch: np.ndarray = field(default_factory=lambda: np.zeros(0))
    yaw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    
    def __init__(
        self,
        time: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        altitude: np.ndarray,
        airspeed: np.ndarray,
        roll: np.ndarray,
        pitch: np.ndarray,
        yaw: np.ndarray,
        angular_velocity: np.ndarray
    ):
        self.time = time
        self.position = position
        self.velocity = velocity
        self.altitude = altitude
        self.airspeed = airspeed
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.angular_velocity = angular_velocity
        
    @property
    def n_samples(self) -> int:
        return int(self.time.shape[0])


class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state, self.eom = self._setup_all_data()
        

    def _setup_all_data(self) -> tuple[StateVector, EquationsOfMotion]:
        state = StateVector(
            position=Vector3(0.0, 0.0, self.config.initial_altitude), 
            velocity=Vector3(self.config.initial_velocity, 0.0, 0.0), 
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(), 
            mass=self.config.mass,
            inertia=np.eye(3)
        )
        eom = EquationsOfMotion(mass=state.mass)

        eom.set_aerodynamic_model(AerodynamicModel(
            reference_area=self.config.reference_area,
            reference_span=self.config.reference_span,
            mean_aerodynamic_chord=self.config.mean_aerodynamic_chord
        ))
        
        state.inertia = eom.estimate_inertia_tensor()

        propulsion = PropulsionSystem(max_thrust=self.config.max_thrust)
        propulsion.set_throttle(1.0) 
        eom.set_propulsion_system(propulsion)
        return state, eom

    @classmethod
    def run_simulation(cls, config: SimulationConfig) -> SimulationData:
        instance = cls(config)
        n = int(round(config.duration / config.dt))
        
        logger.info(f"Starting simulation: duration={config.duration}s, dt={config.dt}s, steps={n}")
        logger.info(f"Initial state: altitude={config.initial_altitude}m, velocity={config.initial_velocity}m/s")
        
        time_history = np.zeros(n)
        position_history = np.zeros((3, n))
        velocity_history = np.zeros((3, n))
        altitude_history = np.zeros(n)
        airspeed_history = np.zeros(n)
        roll_history = np.zeros(n)
        pitch_history = np.zeros(n)
        yaw_history = np.zeros(n)
        angular_velocity_history = np.zeros((3, n))
        
        n_actual = n  
        control_deflections = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}

        for i in range(n):
            altitude = instance.state.position.z
            density = 1.225
            current_time = i * config.dt
            
            if i % 10 == 0 or i < 5:
                roll, pitch, yaw = instance.state.attitude.to_euler()
                alpha, beta = compute_alpha_beta(instance.state.velocity)
                logger.debug(f"Step {i} (t={current_time:.2f}s):")
                logger.debug(f"  Position: {instance.state.position}")
                logger.debug(f"  Velocity: {instance.state.velocity} (mag={instance.state.velocity.magnitude():.2f} m/s)")
                logger.debug(f"  Attitude (deg): roll={np.degrees(roll):.2f}, pitch={np.degrees(pitch):.2f}, yaw={np.degrees(yaw):.2f}")
                logger.debug(f"  Angular velocity: {instance.state.angular_velocity}")
                logger.debug(f"  Alpha (deg): {np.degrees(alpha):.2f}, Beta (deg): {np.degrees(beta):.2f}")
                logger.debug(f"  Control deflections (deg): elevator={np.degrees(control_deflections['elevator']):.2f}, aileron={np.degrees(control_deflections['aileron']):.2f}, rudder={np.degrees(control_deflections['rudder']):.2f}")
            
            try:
                instance.state = instance.eom.integrate(
                    instance.state, config.dt,
                    density=density,
                    control_deflections=control_deflections
                )
            except (OverflowError, RuntimeWarning) as e:
                logger.error(f"Numerical overflow at t={current_time:.1f}s - stopping simulation: {e}")
                n_actual = i
                break
            except Exception as e:
                logger.error(f"Unexpected error at t={current_time:.1f}s: {e}")
                n_actual = i
                break
            
            # Collect data
            time_history[i] = current_time
            position_history[:, i] = instance.state.position.to_array()
            velocity_history[:, i] = instance.state.velocity.to_array()
            altitude_history[i] = altitude
            airspeed_history[i] = instance.state.velocity.magnitude()
            roll, pitch, yaw = instance.state.attitude.to_euler()
            roll_history[i] = roll
            pitch_history[i] = pitch
            yaw_history[i] = yaw
            angular_velocity_history[:, i] = instance.state.angular_velocity.to_array()
            
            # maybe add logger warnings in the future
            # if abs(roll) > np.pi/2:
            
            if i % 100 == 0:
                logger.info(f"t={current_time:.1f}s: alt={altitude:.1f}m, vel={instance.state.velocity.magnitude():.1f}m/s, roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")
        
        logger.info(f"Final state: position={instance.state.position}, velocity={instance.state.velocity}")
        roll, pitch, yaw = instance.state.attitude.to_euler()
        logger.info(f"Final attitude (deg): roll={np.degrees(roll):.2f}, pitch={np.degrees(pitch):.2f}, yaw={np.degrees(yaw):.2f}")
        logger.info(f"Simulation complete!")

        return SimulationData(
            time=time_history,
            position=position_history,
            velocity=velocity_history,
            altitude=altitude_history,
            airspeed=airspeed_history,
            roll=roll_history,
            pitch=pitch_history,
            yaw=yaw_history,
            angular_velocity=angular_velocity_history
        )
    