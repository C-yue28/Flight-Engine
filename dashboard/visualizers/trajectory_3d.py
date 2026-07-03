from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from simulation_runner import SimulationData


class Trajectory3D:

    """ Essentially a rehash of time-series plots except with 3D trajectory """
    
    def __init__(self, ax):
        self.ax = ax
        self.line = None
        self.start_marker = None
        self.end_marker = None
        self.quiver = None
        
    def plot(self, data: SimulationData) -> None:
        self.ax.clear()
        
        self.line, = self.ax.plot(
            data.position[0, :],
            data.position[1, :],
            data.position[2, :],
            'b-', linewidth=2, label='Trajectory'
        )
        
        self.start_marker = self.ax.scatter(
            data.position[0, 0],
            data.position[1, 0],
            data.position[2, 0],
            c='green', s=100, marker='o', label='Start'
        )
        
        self.end_marker = self.ax.scatter(
            data.position[0, -1],
            data.position[1, -1],
            data.position[2, -1],
            c='red', s=100, marker='s', label='End'
        )
        
        # ground visualization - used for debugging altitude sign convention issues
        min_x, max_x = data.position[0, :].min(), data.position[0, :].max()
        min_y, max_y = data.position[1, :].min(), data.position[1, :].max()
        ground_range = max(max_x - min_x, max_y - min_y) / 2 + 100
        
        xx, yy = np.meshgrid(
            np.linspace(data.position[0, :].mean() - ground_range, 
                       data.position[0, :].mean() + ground_range, 10),
            np.linspace(data.position[1, :].mean() - ground_range,
                       data.position[1, :].mean() + ground_range, 10)
        )
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        self.ax.set_xlabel('North (m)')
        self.ax.set_ylabel('East (m)')
        self.ax.set_zlabel('Altitude (m)')
        self.ax.set_title('3D Flight Trajectory')
        self.ax.legend()
        
        self.ax.set_box_aspect([1, 1, 0.5])
        
    def animate(self, data: SimulationData, interval: int = 50) -> FuncAnimation:
        """ todo animate the trajectory """
        pass
