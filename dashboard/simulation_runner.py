from dataclasses import dataclass
from aerodynamics import AerodynamicModel

"""
TODO: Integrate with GUI, add user interactivity, and add data computation
"""

@dataclass
class SimulationConfig:
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


class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.aerodynamic_model = AerodynamicModel(
            config.reference_area,
            config.reference_span,
            config.mean_aerodynamic_chord
        )


class SimulationData:
    
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
        pass