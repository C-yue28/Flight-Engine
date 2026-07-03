import numpy as np
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from core import StateVector, Vector3, Quaternion, compute_alpha_beta
from aerodynamics import AerodynamicModel, AerodynamicCoefficients


"""
File with various unit tests that Claude recommended to ensure the aerodynamics modules funtioned properly
"""


class TestAerodynamicAlignment:
    
    # at zero alpha and beta all rotational parameters should be zero
    def test_zero_alpha_beta_symmetry(self):
        state = StateVector(
            position=Vector3(0.0, 0.0, -1000.0),
            velocity=Vector3(100.0, 0.0, 0.0), 
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(),
            mass=1000.0,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        alpha, beta = compute_alpha_beta(state.velocity)
        assert abs(alpha) < 1e-10
        assert abs(beta) < 1e-10
        
        aero = AerodynamicModel(
            reference_area=20.0,
            reference_span=10.0,
            mean_aerodynamic_chord=2.0
        )
        
        coeffs = aero.compute_coefficients(
            state,
            control_deflections={'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
        )

        assert abs(coeffs.CY) < 1e-9
        assert abs(coeffs.Cl) < 1e-9
        assert abs(coeffs.Cn) < 1e-9
    
    # the only force should be drag in the -x direction with zero angles
    def test_forces_and_moments_alignment(self):
        state = StateVector(
            position=Vector3(0.0, 0.0, -1000.0),
            velocity=Vector3(100.0, 0.0, 0.0),
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(),
            mass=1000.0,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        aero = AerodynamicModel(
            reference_area=20.0,
            reference_span=10.0,
            mean_aerodynamic_chord=2.0
        )
        
        forces, moments = aero.compute_forces_and_moments(
            state,
            control_deflections={'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0},
            density=1.225
        )
        
        assert abs(forces[1]) < 1e-6
        assert abs(moments[0]) < 1e-6
        assert abs(moments[1]) < 1e-6
        assert abs(moments[2]) < 1e-6
    
    # with some sideslip the coefficients should be non zero because of stability moments
    def test_sideslip_induced_moments(self):
        state = StateVector(
            position=Vector3(0.0, 0.0, -1000.0),
            velocity=Vector3(100.0, 5.0, 0.0), 
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(),
            mass=1000.0,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        aero = AerodynamicModel(
            reference_area=20.0,
            reference_span=10.0,
            mean_aerodynamic_chord=2.0
        )
        
        coeffs = aero.compute_coefficients(
            state,
            control_deflections={'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
        )
        assert abs(coeffs.CY) > 1e-6
        assert abs(coeffs.Cl) > 1e-6
        assert abs(coeffs.Cn) > 1e-6
    
    # shifting control deflections should induce moments
    def test_control_surface_effects(self):
        state = StateVector(
            position=Vector3(0.0, 0.0, -1000.0),
            velocity=Vector3(100.0, 0.0, 0.0),
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(),
            mass=1000.0,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        aero = AerodynamicModel(
            reference_area=20.0,
            reference_span=10.0,
            mean_aerodynamic_chord=2.0
        )
        
        coeffs_elevator = aero.compute_coefficients(
            state,
            control_deflections={'elevator': np.radians(10), 'aileron': 0.0, 'rudder': 0.0}
        )
        assert abs(coeffs_elevator.Cm) > 1e-6
        
        coeffs_aileron = aero.compute_coefficients(
            state,
            control_deflections={'elevator': 0.0, 'aileron': np.radians(10), 'rudder': 0.0}
        )
        assert abs(coeffs_aileron.Cl) > 1e-6
        
        coeffs_rudder = aero.compute_coefficients(
            state,
            control_deflections={'elevator': 0.0, 'aileron': 0.0, 'rudder': np.radians(10)}
        )
        assert abs(coeffs_rudder.Cn) > 1e-6


class TestAerodynamicCoefficients:
    
    # at small alpha lift should be linear according to Claude
    def test_lift_slope_linearity(self):
        aero = AerodynamicModel(
            reference_area=20.0,
            reference_span=10.0,
            mean_aerodynamic_chord=2.0
        )
        
        state = StateVector(
            position=Vector3(0.0, 0.0, -1000.0),
            velocity=Vector3(100.0, 0.0, 0.0),
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(),
            mass=1000.0,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        alphas = np.radians([-2, -1, 0, 1, 2])
        CL_values = []
        
        for alpha in alphas:
            state.velocity = Vector3(
                100.0 * np.cos(alpha),
                0.0,
                100.0 * np.sin(alpha)
            )
            
            coeffs = aero.compute_coefficients(state, control_deflections={})
            CL_values.append(coeffs.CL)

        for i in range(len(CL_values) - 1):
            assert CL_values[i + 1] > CL_values[i]
        
        # should be approximately linear
        diffs = np.diff(CL_values)
        assert np.std(diffs) < np.mean(diffs) * 0.5
    
    def test_drag_polar(self):
        aero = AerodynamicModel(
            reference_area=20.0,
            reference_span=10.0,
            mean_aerodynamic_chord=2.0
        )
        
        state = StateVector(
            position=Vector3(0.0, 0.0, -1000.0),
            velocity=Vector3(100.0, 0.0, 0.0),
            attitude=Quaternion.identity(),
            angular_velocity=Vector3.zeros(),
            mass=1000.0,
            inertia=np.diag([1000.0, 2000.0, 1500.0])
        )
        
        alphas = np.radians([0, 5, 10])
        CD_values = []
        CL_values = []
        
        for alpha in alphas:
            state.velocity = Vector3(
                100.0 * np.cos(alpha),
                0.0,
                100.0 * np.sin(alpha)
            )
            
            coeffs = aero.compute_coefficients(state, control_deflections={})
            CD_values.append(coeffs.CD)
            CL_values.append(coeffs.CL)
        
        assert CD_values[1] > CD_values[0]
        assert CD_values[2] > CD_values[1]


class TestLookupTable:
    
    # lookup table needs to be accurate in terms of interpolation
    def test_lookup_table_interpolation(self):
        from aerodynamics import CoefficientLookupTable
        
        alphas = np.radians(np.linspace(-10, 10, 21))
        CL_values = 2 * np.pi * alphas
        
        coeffs_array = np.zeros((len(alphas), 6))
        coeffs_array[:, 0] = CL_values
        
        lookup = CoefficientLookupTable(
            dimensions=['alpha'],
            breakpoints=[alphas],
            coefficients=coeffs_array,
            coefficient_names=['CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn']
        )
        
        test_alpha = np.radians(2.5)
        coeffs = lookup.interpolate(alpha=test_alpha)
        
        expected_CL = 2 * np.pi * test_alpha
        assert np.isclose(coeffs.CL, expected_CL, rtol=0.01)
    
    # some more lookup table testing
    def test_lookup_table_from_function(self):
        from aerodynamics import CoefficientLookupTable
        
        def coefficient_function(alpha):
            CL = 2 * np.pi * alpha
            CD = 0.02 + 0.05 * CL**2
            return AerodynamicCoefficients(CL=CL, CD=CD, CY=0.0, Cl=0.0, Cm=0.0, Cn=0.0)
        
        alphas = np.radians(np.linspace(-10, 10, 11))
        
        lookup = CoefficientLookupTable.from_function(
            dimensions=['alpha'],
            breakpoints=[alphas],
            coefficient_function=coefficient_function
        )
        
        test_alpha = np.radians(5)
        coeffs = lookup.interpolate(alpha=test_alpha)
        
        expected_CL = 2 * np.pi * test_alpha
        expected_CD = 0.02 + 0.05 * expected_CL**2
        
        assert np.isclose(coeffs.CL, expected_CL, rtol=0.01)
        assert np.isclose(coeffs.CD, expected_CD, rtol=0.02)