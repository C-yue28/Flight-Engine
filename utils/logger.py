import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

"""
Centralized logging utilities built with Claude
"""

def setup_simulation_logger(
    log_dir: str = "logs",
    verbose: bool = True,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    handlers = []
    
    if log_to_file:
        sim_handler = logging.FileHandler(log_path / f"simulation_{timestamp}.log")
        sim_handler.setLevel(logging.DEBUG)
        sim_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        sim_handler.setFormatter(sim_formatter)
        handlers.append(sim_handler)
        
        err_handler = logging.FileHandler(log_path / f"errors_{timestamp}.log")
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(sim_formatter)
        handlers.append(err_handler)
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
        force=True 
    )
    
    logger = logging.getLogger("flight_engine")
    logger.info(f"Logging initialized: log_dir={log_dir}, verbose={verbose}")
    
    return logger


def validate_state_vector(
    position,
    velocity,
    attitude,
    angular_velocity,
    logger: Optional[logging.Logger] = None
) -> bool:
    if logger is None:
        logger = logging.getLogger("flight_engine")
    
    valid = True
    
    # Check position
    pos_array = position.to_array() if hasattr(position, 'to_array') else position
    if np.isnan(pos_array).any() or np.isinf(pos_array).any():
        logger.error(f"Invalid position: {position}")
        valid = False
    
    # Check velocity
    vel_array = velocity.to_array() if hasattr(velocity, 'to_array') else velocity
    if np.isnan(vel_array).any() or np.isinf(vel_array).any():
        logger.error(f"Invalid velocity: {velocity}")
        valid = False
    
    # Check attitude
    if hasattr(attitude, 'to_array'):
        att_array = attitude.to_array()
        if np.isnan(att_array).any() or np.isinf(att_array).any():
            logger.error(f"Invalid attitude: {attitude}")
            valid = False
    
    # Check angular velocity
    ang_array = angular_velocity.to_array() if hasattr(angular_velocity, 'to_array') else angular_velocity
    if np.isnan(ang_array).any() or np.isinf(ang_array).any():
        logger.error(f"Invalid angular velocity: {angular_velocity}")
        valid = False
    
    return valid