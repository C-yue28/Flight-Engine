import numpy as np
import logging
from typing import Optional
from core import StateVector, Vector3, compute_alpha_beta, wind_to_body, compute_alpha_beta, Vector3
from .coefficients import AerodynamicCoefficients
from .lookup_table import CoefficientLookupTable

logger = logging.getLogger("flight_engine.aerodynamics")


class AerodynamicModel:

    """
    Reference the documentation for all of the math behind the aerodynamic model
    DISCLAIMER: I am nowhere close to knowing much of anything about aerial physics,
    most of the math/physics is from various sources and AI
    """
    
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
            print("TESTING-----------------------------------------")
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
        
        # Log warnings for extreme conditions
        # if abs(alpha) > np.radians(45):
        #     logger.warning(f"Extreme alpha: {np.degrees(alpha):.2f}°")
        # if abs(beta) > np.radians(45):
        #     logger.warning(f"Extreme beta: {np.degrees(beta):.2f}°")
        # if airspeed < 1.0:
        #     logger.warning(f"Low airspeed: {airspeed:.2f} m/s")

        if airspeed < 1e-6:
            logger.error("Zero airspeed in coefficient computation")
            return AerodynamicCoefficients(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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
        Simple linear/quasi-linear aerodynamic model with smoothed stall
        transition and a first-order compressibility correction.

        Sign convention: positive elevator = trailing-edge-down (nose-up
        moment expected to be negative Cm_elevator); positive aileron =
        right-aileron-down / left-up (positive roll rate expected);
        positive rudder = trailing-edge-left (positive Cn expected).
        """
        # --- Guard against non-finite inputs (e.g. V -> 0 upstream) ---
        inputs = [alpha, beta, p_hat, q_hat, r_hat, mach]
        if not all(np.isfinite(x) for x in inputs):
            logger.error("Non-finite input to _simple_coefficients; returning zero coefficients")
            return AerodynamicCoefficients(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # --- Clamp control deflections to the linear model's valid range ---
        max_deflection = np.radians(25.0)
        elevator = np.clip(control_deflections.get('elevator', 0.0), -max_deflection, max_deflection)
        aileron = np.clip(control_deflections.get('aileron', 0.0), -max_deflection, max_deflection)
        rudder = np.clip(control_deflections.get('rudder', 0.0), -np.radians(30.0), np.radians(30.0))

        # --- Compressibility correction (Prandtl-Glauert, subsonic only) ---
        mach_clamped = np.clip(mach, 0.0, 0.85)
        if mach_clamped > 0.85:
            logger.warning(f"Mach {mach:.2f} outside valid range for Prandtl-Glauert correction; clamped to 0.85")
        beta_pg = np.sqrt(max(1.0 - mach_clamped ** 2, 1e-6))

        CL_0 = 0.3
        CL_alpha = 2 * np.pi / beta_pg
        CL_elevator = 0.4

        # --- Smoothed stall transition (tanh blend instead of hard switch) ---
        alpha_stall = np.radians(15)
        stall_width = np.radians(3)   # transition width, tune to taste
        blend = 0.5 * (1.0 - np.tanh((abs(alpha) - alpha_stall) / stall_width))
        CL_linear = CL_0 + CL_alpha * alpha
        CL_stalled = (CL_0 + CL_alpha * alpha_stall * np.sign(alpha)) * 0.8
        CL = blend * CL_linear + (1.0 - blend) * CL_stalled + CL_elevator * elevator

        CD_0 = 0.02
        k = 0.05
        CD_elevator = 0.05
        CD = CD_0 + k * CL ** 2 + CD_elevator * abs(elevator)

        CY_beta = -0.5
        CY_rudder = 0.3
        CY = CY_beta * beta + CY_rudder * rudder

        Cl_p = -0.5
        Cl_beta = -0.1
        Cl_aileron = 0.2
        Cl_rudder = 0.01   # small rudder-induced roll
        Cl = Cl_p * p_hat + Cl_beta * beta + Cl_aileron * aileron + Cl_rudder * rudder

        Cm_0 = 0.0
        Cm_alpha = -0.8
        Cm_q = -8.0
        Cm_elevator = -1.2
        Cm = Cm_0 + Cm_alpha * alpha + Cm_q * q_hat + Cm_elevator * elevator

        # Cn_beta corrected to positive: restoring (weathercock-stable) yaw moment
        Cn_beta = 0.1
        Cn_r = -0.1
        Cn_rudder = 0.15
        Cn_aileron = -0.02   # adverse yaw
        Cn = Cn_beta * beta + Cn_r * r_hat + Cn_rudder * rudder + Cn_aileron * aileron

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
        F_body = wind_to_body(F_wind, alpha, beta)  # Use positive beta
        
        L = coeffs.Cl * q * self.reference_area * self.reference_span
        M = coeffs.Cm * q * self.reference_area * self.mean_aerodynamic_chord
        N = coeffs.Cn * q * self.reference_area * self.reference_span
        
        return F_body.to_array(), np.array([L, M, N], dtype=np.float64)
