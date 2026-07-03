import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import csv

from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from .panels import ConfigPanel
from .plot_panel import PlotPanel
from simulation_runner import SimulationRunner

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flight Simulation Dashboard")
        self.geometry("1200x750")

        self.data = None
        
        self._setup_layout()
        self._create_interaction_points()

    def _setup_layout(self) -> None:
        container = ttk.Panedwindow(self, orient="horizontal")
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container, width=280)
        right = ttk.Frame(container)
        container.add(left, weight=0)
        container.add(right, weight=1)

        self.config_panel = ConfigPanel(left)
        self.config_panel.pack(fill="both", expand=True)

        toolbar = ttk.Frame(right)
        toolbar.pack(side="top", fill="x", padx=5, pady=(5, 0))
        self.animate_button = ttk.Button(toolbar, text="▶ Animate Trajectory")
        self.animate_button.pack(side="left")
        self.animate_button.state(["disabled"])

        self.status_var = tk.StringVar(value="Configure parameters and click Run Simulation.")
        ttk.Label(toolbar, textvariable=self.status_var, foreground="gray").pack(side="left", padx=12)

        self.plot_panel = PlotPanel(right)
        self.plot_panel.pack(fill="both", expand=True, padx=5, pady=5)

    def _create_interaction_points(self) -> None:
        self.config_panel.run_button.configure(command=self._run_sim)
        self.config_panel.export_button.configure(command=self._export_data)
        self.animate_button.configure(command=self.plot_panel.animate_trajectory)

    def _run_sim(self) -> None:
        self.status_var.set("Running simulation…")
        self.update_idletasks()

        config = self.config_panel._get_cfg()
        self.data = SimulationRunner.run_simulation(config)

        # Update plots with new data
        self.plot_panel.update_plots(self.data)
        self.animate_button.state(["!disabled"])
        self.config_panel.export_button.state(["!disabled"])

        self.status_var.set(f"Simulation complete -> {self.data.n_samples} samples")

    def _export_data(self) -> bool:
        if self.data is None:
            return False
        
        filepath = filedialog.asksaveasfilename(
            title="Export simulation data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if not filepath:
            return False

        with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time_s", "x_north_m", "y_east_m", "z_down_m",
                    "u_mps", "v_mps", "w_mps",
                    "altitude_m", "airspeed_mps",
                    "roll_rad", "pitch_rad", "yaw_rad",
                    "p_radps", "q_radps", "r_radps",
                ])
                for i in range(self.data.n_samples):
                    writer.writerow([
                        self.data.time[i],
                        self.data.position[0, i], self.data.position[1, i], self.data.position[2, i],
                        self.data.velocity[0, i], self.data.velocity[1, i], self.data.velocity[2, i],
                        self.data.altitude[i], self.data.airspeed[i],
                        self.data.roll[i], self.data.pitch[i], self.data.yaw[i],
                        self.data.angular_velocity[0, i], self.data.angular_velocity[1, i], self.data.angular_velocity[2, i],
                    ])

        self.status_var.set(f"Exported {self.data.n_samples} samples to {filepath}")
        return True
