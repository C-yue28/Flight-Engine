from dataclasses import dataclass
import numpy as np

# wrapper for all aerodynamic coefficients

@dataclass
class AerodynamicCoefficients:
    """
    Lift, drag, side, rolling, pitching, yawing coefficients
    """
    CL: float = 0.0
    CD: float = 0.0
    CY: float = 0.0
    Cl: float = 0.0
    Cm: float = 0.0
    Cn: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.CL, self.CD, self.CY, self.Cl, self.Cm, self.Cn], dtype=np.float64)
    
    @classmethod
    def zeros(cls) -> 'AerodynamicCoefficients':
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def copy(self) -> 'AerodynamicCoefficients':
        return AerodynamicCoefficients(
            self.CL, self.CD, self.CY, self.Cl, self.Cm, self.Cn
        )
