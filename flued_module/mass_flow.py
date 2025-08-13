"""
Mass flow and pressure calculations for chimney simulation
Based on algorithms from FLUESIM5.BAS mass.flow and pres.param subroutines
"""
import numpy as np
from .physics import PhysicalProperties

class MassFlow:
    """Mass flow and pressure calculations for flue systems"""
    
    def __init__(self):
        self.physics = PhysicalProperties()
    
    def buoyancy_pressure(self, height, density_cold, density_hot):
        """Calculate buoyancy pressure difference
        
        Args:
            height: Height difference (m)
            density_cold: Density of cold fluid (kg/m³)
            density_hot: Density of hot fluid (kg/m³)
            
        Returns:
            Buoyancy pressure (Pa)
        """
        return (density_cold - density_hot) * self.physics.GRAVITY * height
    
    def pressure_loss_friction(self, friction_factor, length, diameter, density, velocity):
        """Calculate pressure loss due to friction
        
        Args:
            friction_factor: Darcy friction factor (dimensionless)
            length: Pipe length (m)
            diameter: Pipe diameter (m)
            density: Fluid density (kg/m³)
            velocity: Flow velocity (m/s)
            
        Returns:
            Pressure loss (Pa)
        """
        return friction_factor * (length / diameter) * (density * velocity**2) / 2
    
    def pressure_loss_minor(self, loss_coefficient, density, velocity):
        """Calculate minor pressure losses
        
        Args:
            loss_coefficient: Minor loss coefficient (dimensionless)
            density: Fluid density (kg/m³)
            velocity: Flow velocity (m/s)
            
        Returns:
            Pressure loss (Pa)
        """
        return loss_coefficient * (density * velocity**2) / 2
    
    def entrance_loss_coefficient(self, geometry='sharp'):
        """Get entrance loss coefficient
        
        Args:
            geometry: 'sharp', 'rounded', or 'bell_mouth'
            
        Returns:
            Loss coefficient (dimensionless)
        """
        coefficients = {
            'sharp': 0.5,
            'rounded': 0.1,
            'bell_mouth': 0.05
        }
        return coefficients.get(geometry, 0.5)
    
    def exit_loss_coefficient(self):
        """Get exit loss coefficient
        
        Returns:
            Loss coefficient (dimensionless) - typically 1.0
        """
        return 1.0
    
    def elbow_loss_coefficient(self, angle_degrees, diameter, radius_curvature=None):
        """Calculate elbow loss coefficient
        
        Args:
            angle_degrees: Bend angle in degrees
            diameter: Pipe diameter (m)
            radius_curvature: Radius of curvature (m), if None uses 1.5*diameter
            
        Returns:
            Loss coefficient (dimensionless)
        """
        if radius_curvature is None:
            radius_curvature = 1.5 * diameter
        
        # Ratio of radius to diameter
        r_over_d = radius_curvature / diameter
        
        # Base loss coefficient for 90-degree bend
        if r_over_d >= 2:
            K_90 = 0.2
        elif r_over_d >= 1:
            K_90 = 0.35
        else:
            K_90 = 0.5
        
        # Scale for actual angle
        angle_factor = (angle_degrees / 90) ** 1.5
        
        return K_90 * angle_factor
    
    def calculate_mass_flow_rate(self, driving_pressure, total_resistance, density):
        """Calculate mass flow rate from driving pressure and system resistance
        
        Args:
            driving_pressure: Net driving pressure (Pa)
            total_resistance: Total system flow resistance (Pa·s²/kg²)
            density: Average fluid density (kg/m³)
            
        Returns:
            Mass flow rate (kg/s)
        """
        if driving_pressure <= 0 or total_resistance <= 0:
            return 0.001  # Minimum flow to avoid division by zero
        
        # From pressure balance: dp = resistance * (mass_flow / area)²
        # Solve for mass flow rate
        mass_flow = np.sqrt(abs(driving_pressure) / total_resistance)
        
        return mass_flow
    
    def system_resistance_analysis(self, segments_data):
        """Calculate total system flow resistance
        
        Args:
            segments_data: List of dictionaries with segment properties
            Each dict should contain: length, diameter, friction_factor, minor_losses
            
        Returns:
            Total flow resistance (Pa·s²/kg²)
        """
        total_resistance = 0.0
        
        for segment in segments_data:
            length = segment['length']
            diameter = segment['diameter']
            friction_factor = segment['friction_factor']
            minor_losses = segment.get('minor_losses', [])
            
            # Area
            area = np.pi * (diameter / 2) ** 2
            
            # Friction resistance
            friction_resistance = friction_factor * length / (diameter * area**2)
            
            # Minor loss resistance
            minor_resistance = sum(minor_losses) / area**2
            
            # Total segment resistance
            segment_resistance = friction_resistance + minor_resistance
            total_resistance += segment_resistance
        
        return total_resistance
    
    def draft_calculation(self, chimney_height, T_gas_avg, T_ambient, pressure_ambient):
        """Calculate natural draft pressure
        
        Args:
            chimney_height: Total chimney height (m)
            T_gas_avg: Average gas temperature in chimney (K)
            T_ambient: Ambient temperature (K)
            pressure_ambient: Ambient pressure (Pa)
            
        Returns:
            Draft pressure (Pa)
        """
        # Densities
        rho_ambient = self.physics.air_density(T_ambient, pressure_ambient)
        rho_gas = self.physics.air_density(T_gas_avg, pressure_ambient)
        
        # Draft pressure
        draft = self.buoyancy_pressure(chimney_height, rho_ambient, rho_gas)
        
        return draft
    
    def wind_pressure_effect(self, wind_speed, wind_direction_deg, chimney_height, 
                           building_height=None, pressure_coefficients=None):
        """Calculate wind pressure effects on chimney draft
        
        Args:
            wind_speed: Wind speed (m/s)
            wind_direction_deg: Wind direction relative to building (degrees)
            chimney_height: Chimney height above ground (m)
            building_height: Building height (m), default is chimney_height * 0.5
            pressure_coefficients: Dict with 'windward' and 'leeward' coefficients
            
        Returns:
            Wind pressure difference (Pa)
        """
        if building_height is None:
            building_height = chimney_height * 0.5
        
        if pressure_coefficients is None:
            pressure_coefficients = {'windward': 0.8, 'leeward': -0.5}
        
        # Dynamic pressure
        rho_air = self.physics.air_density(288.15)  # Standard air density
        q_wind = 0.5 * rho_air * wind_speed**2
        
        # Height factor for wind speed variation
        if chimney_height > building_height:
            height_factor = (chimney_height / building_height) ** 0.15
        else:
            height_factor = 1.0
        
        # Pressure difference due to wind
        delta_p_wind = q_wind * height_factor * (
            pressure_coefficients['windward'] - pressure_coefficients['leeward']
        )
        
        # Direction factor (simplified)
        direction_factor = abs(np.cos(np.radians(wind_direction_deg)))
        
        return delta_p_wind * direction_factor
    
    def flow_network_solver(self, nodes, connections, boundary_conditions, max_iterations=50, tolerance=1e-6):
        """Solve flow network using iterative method
        
        Args:
            nodes: Dictionary of node properties
            connections: List of connection dictionaries with flow resistance
            boundary_conditions: Dictionary of known pressures or flows
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for pressure residuals
            
        Returns:
            Dictionary with solved pressures and flow rates
        """
        # This is a simplified network solver
        # In practice, this would implement the full matrix solution as in FLUESIM5
        
        n_nodes = len(nodes)
        n_connections = len(connections)
        
        # Initialize pressures and flows
        pressures = np.zeros(n_nodes)
        flows = np.zeros(n_connections)
        
        # Apply boundary conditions for pressures
        for node_id, pressure in boundary_conditions.get('pressures', {}).items():
            if node_id < n_nodes:
                pressures[node_id] = pressure
        
        # Iterative solution
        for iteration in range(max_iterations):
            old_pressures = pressures.copy()
            
            # Update flows based on current pressures
            for i, conn in enumerate(connections):
                node_from = conn['from_node']
                node_to = conn['to_node']
                resistance = conn['resistance']
                
                dp = pressures[node_from] - pressures[node_to]
                flows[i] = np.sign(dp) * np.sqrt(abs(dp) / resistance) if resistance > 0 else 0
            
            # Update pressures based on flow continuity
            # (Simplified - full implementation would use matrix methods)
            for i in range(n_nodes):
                if i not in boundary_conditions.get('pressures', {}):
                    # Sum flows in and out of node
                    flow_sum = 0
                    for j, conn in enumerate(connections):
                        if conn['to_node'] == i:
                            flow_sum += flows[j]
                        elif conn['from_node'] == i:
                            flow_sum -= flows[j]
                    
                    # Pressure correction (simplified)
                    pressures[i] += 0.1 * flow_sum
            
            # Check convergence
            residual = np.max(np.abs(pressures - old_pressures))
            if residual < tolerance:
                break
        
        return {
            'pressures': pressures,
            'flows': flows,
            'iterations': iteration + 1,
            'converged': residual < tolerance
        }
    
    def equivalent_leakage_area(self, flow_rate, pressure_difference, reference_pressure=10.0, exponent=0.5):
        """Calculate equivalent leakage area
        
        Args:
            flow_rate: Flow rate (m³/s)
            pressure_difference: Pressure difference (Pa)
            reference_pressure: Reference pressure for ELA calculation (Pa)
            exponent: Flow exponent (typically 0.5 for turbulent flow)
            
        Returns:
            Equivalent leakage area (m²)
        """
        if pressure_difference <= 0:
            return 0.0
        
        # Standard air density
        rho = self.physics.air_density(293.15)  # 20°C
        
        # Flow coefficient
        C = flow_rate / (pressure_difference ** exponent)
        
        # ELA at reference pressure
        reference_flow = C * (reference_pressure ** exponent)
        ela = reference_flow / np.sqrt(2 * reference_pressure / rho)
        
        return ela