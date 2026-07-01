import numpy as np
from typing import List, Tuple, Optional, Callable
from .coefficients import AerodynamicCoefficients


class CoefficientLookupTable:
    """
    Lookup table for aerodynamic coefficients w/ linear interpolation
    Essentially it just precomputes a bunch of coefficient data, and then
    interpolates between them in a really cool way so that it can calculate the coefficients
    at any given point
    """
    
    def __init__(
        self,
        dimensions: List[str],
        breakpoints: List[np.ndarray],
        coefficients: np.ndarray,
        coefficient_names: List[str] = ['CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn']
    ):
        self.dimensions = dimensions
        self.breakpoints = breakpoints
        self.coefficients = coefficients
        self.coefficient_names = coefficient_names
    
    @classmethod
    def from_function(
        cls,
        dimensions: List[str],
        breakpoints: List[np.ndarray],
        coefficient_function: Callable,
        coefficient_names: List[str] = ['CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn']
    ) -> 'CoefficientLookupTable':
        meshes = np.meshgrid(*breakpoints, indexing='ij')
        
        shape = tuple(len(bp) for bp in breakpoints) + (len(coefficient_names),)
        coeffs = np.zeros(shape, dtype=np.float64)
        
        for indices in np.ndindex(shape[:-1]):
            dim_values = [meshes[i][indices] for i in range(len(dimensions))]
            result = coefficient_function(*dim_values)
            coeffs[indices] = result.to_array()
        
        return cls(dimensions, breakpoints, coeffs, coefficient_names)
    
    # Some heavy AI use here
    def interpolate(self, **kwargs) -> AerodynamicCoefficients:
        query = []
        for dim in self.dimensions:
            if dim not in kwargs:
                raise ValueError(f"Missing value for dimension: {dim}")
            query.append(kwargs[dim])
        
        query = np.array(query)
        
        indices = []
        weights = []
        
        for i, (value, bp) in enumerate(zip(query, self.breakpoints)):
            idx = np.searchsorted(bp, value)
            
            if idx == 0:
                indices.append([0, 0])
                weights.append([1.0, 0.0])
            elif idx >= len(bp):
                indices.append([len(bp) - 1, len(bp) - 1])
                weights.append([1.0, 0.0])
            else:
                lower = idx - 1
                upper = idx
                
                indices.append([lower, upper])
                
                # Calculate interpolation weight
                if bp[upper] != bp[lower]:
                    w = (value - bp[lower]) / (bp[upper] - bp[lower])
                else:
                    w = 0.5
                weights.append([1.0 - w, w])
        
        result = np.zeros(len(self.coefficient_names), dtype=np.float64)
        total_weight = 0.0
        
        # Iterate over all corner points of the "hypercube" 
        # What even is a hypercube

        from itertools import product
        for corner in product([0, 1], repeat=len(self.dimensions)):
            # Get indices and weight for this corner
            idx_tuple = tuple(indices[i][corner[i]] for i in range(len(self.dimensions)))
            weight = np.prod([weights[i][corner[i]] for i in range(len(self.dimensions))])
            
            # Add weighted contribution
            result += weight * self.coefficients[idx_tuple]
            total_weight += weight
        
        if total_weight > 0:
            result /= total_weight
        
        return AerodynamicCoefficients(*result)
    
    def get_coefficient_array(self, coefficient_name: str) -> np.ndarray:
        idx = self.coefficient_names.index(coefficient_name)
        return self.coefficients[..., idx]
