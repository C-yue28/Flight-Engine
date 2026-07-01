import numpy as np
from typing import Optional
from core import StateVector, Vector3, compute_alpha_beta, wind_to_body, compute_alpha_beta, Vector3
from .coefficients import AerodynamicCoefficients
from .lookup_table import CoefficientLookupTable


class AerodynamicModel:
    
    def __init__(
        self,
        reference_area: float = 20.0,
        reference_span: float = 10.0,
        mean_aerodynamic_chord: float = 2.0,
        lookup_table: Optional[CoefficientLookupTable] = None
    ):
        self.reference_area = reference_area
        self.reference_span = reference_span
        self.mean_aerodynamic_chord = mean_aerodynamic_chord
        self.lookup_table = lookup_table
        
        if lookup_table is None:
            self._use_simple_model = True
        else:
            self._use_simple_model = False
    
    def compute_coefficients(
        self,
        state: StateVector,
        control_deflections: Optional[dict] = None,
        mach: float = 0.0,
        reynolds: float = 1e6
    ) -> AerodynamicCoefficients:
        if control_deflections is None:
            control_deflections = {}

        # angles, angular rates, control deflections
        alpha, beta = compute_alpha_beta(state.velocity)
        
        p = state.angular_velocity.x
        q = state.angular_velocity.y
        r = state.angular_velocity.z
        
        airspeed = state.velocity.magnitude()
        if airspeed < 1e-6:
            return AerodynamicCoefficients.zeros()
        
        p_hat = p * self.reference_span / (2 * airspeed)
        q_hat = q * self.mean_aerodynamic_chord / (2 * airspeed)
        r_hat = r * self.reference_span / (2 * airspeed)
        
        if self._use_simple_model:
            return self._simple_coefficients(alpha, beta, p_hat, q_hat, r_hat, 
                                            control_deflections, mach)
        else:
            return self.lookup_table.interpolate(
                alpha=alpha,
                beta=beta,
                mach=mach,
                reynolds=reynolds,
                **control_deflections
            )
    
    def _simple_coefficients(
        self,
        alpha: float,
        beta: float,
        p_hat: float,
        q_hat: float,
        r_hat: float,
        control_deflections: dict,
        mach: float
    ) -> AerodynamicCoefficients:
        """
        Simple linear aerodynamic model, for making sure that the base framework works so that debugging the more complex model becomes easier
        """
        elevator = control_deflections.get('elevator', 0.0)
        aileron = control_deflections.get('aileron', 0.0)
        rudder = control_deflections.get('rudder', 0.0)
        
        CL = 2 * np.pi * alpha + 0.3 * elevator
        CD = 0.02 + 0.05 * CL**2 + 0.1 * abs(elevator)
        CY = -0.5 * beta + 0.3 * rudder
        Cl = -0.5 * p_hat - 0.1 * beta + 0.2 * aileron
        Cl = -0.5 * p_hat - 0.1 * beta + 0.2 * aileron    
        Cm = -0.5 * alpha - 0.5 * q_hat - 0.8 * elevator  
        Cn = -0.1 * beta - 0.1 * r_hat + 0.15 * rudder
        
        return AerodynamicCoefficients(CL, CD, CY, Cl, Cm, Cn)
    
    def compute_forces_and_moments(
        self,
        state: StateVector,
        control_deflections: Optional[dict] = None,
        density: float = 1.225,
        mach: float = 0.0,
        reynolds: float = 1e6
    ) -> tuple[np.ndarray, np.ndarray]:
        coeffs = self.compute_coefficients(state, control_deflections, mach, reynolds)
        
        airspeed = state.velocity.magnitude()
        q = 0.5 * density * airspeed**2
        
        alpha, beta = compute_alpha_beta(state.velocity)
        
        # drag operates in the negative direction relative to the wind frame 
        # (i.e. the forward wind direction is the backwards plane direction)
        F_drag = -coeffs.CD * q * self.reference_area
        F_lift = coeffs.CL * q * self.reference_area
        F_side = coeffs.CY * q * self.reference_area
        
        F_wind = Vector3(F_drag, F_side, -F_lift) # air goes down, plane comes up
        F_body = wind_to_body(F_wind, alpha, -beta)
        
        L = coeffs.Cl * q * self.reference_area * self.reference_span
        M = coeffs.Cm * q * self.reference_area * self.mean_aerodynamic_chord
        N = coeffs.Cn * q * self.reference_area * self.reference_span
        
        return F_body.to_array(), np.array([L, M, N], dtype=np.float64)
