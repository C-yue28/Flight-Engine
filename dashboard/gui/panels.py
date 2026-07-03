import tkinter as tk
from tkinter import ttk
# from typing import 

from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from simulation_runner import SimulationConfig

FIELDS = [
    ("initial_altitude", "Initial altitude", "m"),
    ("initial_velocity", "Initial velocity", "m/s"),
    ("duration", "Duration", "s"),
    ("dt", "Time step (dt)", "s"),
    ("mass", "Mass", "kg"),
    ("reference_area", "Reference area", "m²"),
    ("reference_span", "Reference span", "m"),
    ("mean_aerodynamic_chord", "Mean aero. chord", "m"),
    ("max_thrust", "Max thrust", "N"),
]

class ConfigPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.entries = {}
        self._error = tk.StringVar(value="")
        self._build_ui()
    
    def _build_ui(self) -> None:
        ttk.Label(self, text="Simulation Parameters", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        row = 1
        for field_name, label, unit in FIELDS:
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(self, textvariable=var, width=12)
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            ttk.Label(self, text=unit, foreground="gray").grid(row=row, column=2, sticky="w")
            self.entries[field_name] = var
            row += 1

        self._error_label = ttk.Label(self, textvariable=self._error, foreground="red", wraplength=220)
        self._error_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 6))
        row += 1

        button_frame = ttk.Frame(self)
        button_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        button_frame.columnconfigure((0, 1, 2), weight=1)

        self.run_button = ttk.Button(button_frame, text="Run Simulation")
        self.run_button.grid(row=0, column=0, padx=2, sticky="ew")

        # self.reset_button = ttk.Button(button_frame, text="Reset", command=self.set_defaults)
        # self.reset_button.grid(row=0, column=1, padx=2, sticky="ew")

        self.export_button = ttk.Button(button_frame, text="Export Data")
        self.export_button.grid(row=0, column=2, padx=2, sticky="ew")
        self.export_button.state(["disabled"])  # enabled once data exists

        self.columnconfigure(1, weight=1)
    
    # getter for simulation config
    def _get_cfg(self) -> SimulationConfig:
        kwargs = {}
        for field_name, desc, unit in FIELDS:
            raw = self.entries[field_name].get()
            try:
                kwargs[field_name] = float(raw)
            except ValueError:
                raise ValueError("All fields must be numerical")
        return SimulationConfig(**kwargs)
    
    def _set_cfg(self, config: SimulationConfig) -> None:
        for field_name, desc, units in FIELDS:
            self.entries[field_name].set(str(getattr(config, field_name)))
