from tkinter import ttk

from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from simulation_runner import SimulationData
from visualizers.trajectory_3d import Trajectory3D
from visualizers.time_series import TimeSeriesPlots


class PlotPanel(ttk.Frame):
    
    def __init__(self, parent):
        super().__init__(parent)
        self.data = None
        self.trajectory_viz = None
        self.timeseries_viz = None
        self.current_animation = None
        
        self._setup_layout()
        
    def _setup_layout(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        
        self.trajectory_frame = ttk.Frame(self.notebook)
        self.timeseries_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.trajectory_frame, text="3D Trajectory")
        self.notebook.add(self.timeseries_frame, text="Time Series")
        
        self.fig_3d = Figure(figsize=(8, 6))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, self.trajectory_frame)
        self.canvas_3d.get_tk_widget().pack(fill="both", expand=True)
        
        toolbar_3d = NavigationToolbar2Tk(self.canvas_3d, self.trajectory_frame)
        toolbar_3d.update()
        toolbar_3d.pack(side="bottom", fill="x")
        
        self.fig_2d = Figure(figsize=(10, 8))
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, self.timeseries_frame)
        self.canvas_2d.get_tk_widget().pack(fill="both", expand=True)
        
        toolbar_2d = NavigationToolbar2Tk(self.canvas_2d, self.timeseries_frame)
        toolbar_2d.update()
        toolbar_2d.pack(side="bottom", fill="x")
        
        self.trajectory_viz = Trajectory3D(self.ax_3d)
        self.timeseries_viz = TimeSeriesPlots(self.fig_2d)
        
    def update_plots(self, data: SimulationData) -> None:
        self.data = data

        self.trajectory_viz.plot(data)
        self.canvas_3d.draw()
        self.timeseries_viz.plot(data)
        self.canvas_2d.draw()
        
    def animate_trajectory(self) -> None:
        """ todo animate stuff """
        pass
        
    def animate_timeseries(self) -> None:
        """ todo animate stuff"""
        pass
