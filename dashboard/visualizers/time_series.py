from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from simulation_runner import SimulationData


class TimeSeriesPlots:

    """ 
    6 time series plots:
    - Altitude
    - Airspeed
    - Roll
    - Pitch
    - Yaw
    - Angular velocity
    """
    
    def __init__(self, fig):
        self.fig = fig
        self.axes = None
        self.lines = {}
        
    def plot(self, data: SimulationData) -> None:
        self.fig.clear()
        
        self.axes = self.fig.subplots(2, 3)
        self.axes = self.axes.flatten()
        
        self._plot_altitude(data, self.axes[0])
        self._plot_airspeed(data, self.axes[1])
        self._plot_roll(data, self.axes[2])
        self._plot_pitch(data, self.axes[3])
        self._plot_yaw(data, self.axes[4])
        self._plot_angular_velocity(data, self.axes[5])
        
        self.fig.tight_layout()
        
    def _plot_altitude(self, data: SimulationData, ax) -> None:
        ax.plot(data.time, data.altitude, 'b-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title('Altitude')
        ax.grid(True, alpha=0.3)
        self.lines['altitude'] = ax.lines[0]
        
    def _plot_airspeed(self, data: SimulationData, ax) -> None:
        ax.plot(data.time, data.airspeed, 'g-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Airspeed (m/s)')
        ax.set_title('Airspeed')
        ax.grid(True, alpha=0.3)
        self.lines['airspeed'] = ax.lines[0]
        
    def _plot_roll(self, data: SimulationData, ax) -> None:
        ax.plot(data.time, np.degrees(data.roll), 'r-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Roll (deg)')
        ax.set_title('Roll Angle')
        ax.grid(True, alpha=0.3)
        self.lines['roll'] = ax.lines[0]
        
    def _plot_pitch(self, data: SimulationData, ax) -> None:
        ax.plot(data.time, np.degrees(data.pitch), 'm-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pitch (deg)')
        ax.set_title('Pitch Angle')
        ax.grid(True, alpha=0.3)
        self.lines['pitch'] = ax.lines[0]
        
    def _plot_yaw(self, data: SimulationData, ax) -> None:
        ax.plot(data.time, np.degrees(data.yaw), 'c-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw (deg)')
        ax.set_title('Yaw Angle')
        ax.grid(True, alpha=0.3)
        self.lines['yaw'] = ax.lines[0]
        
    def _plot_angular_velocity(self, data: SimulationData, ax) -> None:
        angular_vel_mag = np.linalg.norm(data.angular_velocity, axis=0)
        ax.plot(data.time, np.degrees(angular_vel_mag), 'k-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (deg/s)')
        ax.set_title('Angular Velocity Magnitude')
        ax.grid(True, alpha=0.3)
        self.lines['angular_velocity'] = ax.lines[0]
        
    def animate(self, data: SimulationData, interval: int = 50) -> FuncAnimation:
        """ TODO ANIMATE """
        pass
