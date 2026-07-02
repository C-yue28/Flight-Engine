import tkinter as tk

"""
Todo: Build dual-panel layout for the main window and stuff in panels.py
"""

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flight Simulation Dashboard")
        self.geometry("800x600")
        
        self._setup_layout()

    def _setup_layout(self):
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="Flight Simulation Dashboard", font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        self.status_label = tk.Label(main_frame, text="Ready to start simulation", font=("Arial", 10))
        self.status_label.pack(pady=10)
        
        start_button = tk.Button(main_frame, text="Start Simulation", command=self.start_simulation)
        start_button.pack(pady=20)

    def _create_buttons(self):
        pass

