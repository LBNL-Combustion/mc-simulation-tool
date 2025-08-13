# """
# Energy conservation diagnostics for chimney simulation
# """
# import numpy as np
# from typing import Dict, Any

# class EnergyDiagnostics:
#     """
#     Checks energy conservation and physical bounds at each timestep
#     """
    
#     def __init__(self):
#         self.warnings = []
#         self.errors = []
        
#     def check_segment_energy_balance(self, segment_data: Dict[str, Any], 
#                                    dt: float, results: Dict[str, Any]) -> Dict[str, float]:
#         """
#         Check if energy changes respect physical limits
        
#         Args:
#             segment_data: Input data for segment
#             dt: Time step
#             results: Heat transfer results
            
#         Returns:
#             Dictionary with diagnostic values
#         """
#         diagnostics = {}
        
#         # Extract values
#         T_gas = segment_data['gas_temperature']
#         T_ambient = segment_data['ambient_temperature']
#         diameter = segment_data['diameter']
#         length = segment_data['length']
#         pressure = segment_data['pressure']
        
#         # Gas properties
#         cp_gas = 1000.0  # Approximate, J/(kg·K)
#         rho_gas = pressure / (287.0 * T_gas)
#         gas_volume = np.pi * (diameter/2)**2 * length
#         gas_mass = rho_gas * gas_volume
        
#         # Available thermal energy in gas (above ambient)
#         available_energy = gas_mass * cp_gas * (T_gas - T_ambient)
#         diagnostics['available_energy_J'] = available_energy
        
#         # Heat transfer rates
#         h_inner = results.get('h_inner', 0)
#         area = np.pi * diameter * length
        
#         # Maximum possible heat loss (if wall at ambient)
#         max_q_loss = h_inner * area * (T_gas - T_ambient)
#         diagnostics['max_heat_loss_W'] = max_q_loss
        
#         # Maximum cooling in one timestep
#         if gas_mass > 0:
#             max_dT = available_energy / (gas_mass * cp_gas)
#             diagnostics['max_cooling_K'] = max_dT
            
#             # Time to cool to ambient
#             if max_q_loss > 0:
#                 time_to_ambient = available_energy / max_q_loss
#                 diagnostics['time_to_ambient_s'] = time_to_ambient
                
#                 # Check if timestep is too large
#                 if dt > 0.1 * time_to_ambient and T_gas > T_ambient + 1:
#                     self.warnings.append(
#                         f"Timestep {dt}s may be too large. "
#                         f"Time to cool to ambient: {time_to_ambient:.1f}s"
#                     )
        
#         # Check for violations
#         if T_gas < T_ambient - 0.1:
#             self.errors.append(
#                 f"Gas temperature {T_gas-273.15:.1f}°C below ambient {T_ambient-273.15:.1f}°C"
#             )
            
#         # Check wall temperatures
#         if 'wall_temperature_new' in results:
#             T_wall = results['wall_temperature_new']
#             if T_wall < T_ambient - 0.1:
#                 self.errors.append(
#                     f"Wall temperature {T_wall-273.15:.1f}°C below ambient"
#                 )
                
#         return diagnostics
    
#     def check_condensation_physics(self, T_gas: float, T_wall: float, 
#                                   T_ambient: float, condensation_rate: float) -> None:
#         """
#         Verify condensation only occurs under correct conditions
#         """
#         # Condensation should only add heat if moisture is present
#         # and wall is below dew point
#         if condensation_rate > 0 and T_wall >= T_gas:
#             self.errors.append(
#                 "Condensation occurring when wall is hotter than gas!"
#             )
            
#         if T_wall < T_ambient and condensation_rate > 0:
#             self.warnings.append(
#                 "Wall below ambient but condensation occurring - check physics"
#             )
    
#     def get_report(self) -> str:
#         """Get diagnostic report"""
#         report = []
        
#         if self.errors:
#             report.append("ERRORS:")
#             for error in self.errors:
#                 report.append(f"  - {error}")
                
#         if self.warnings:
#             report.append("\nWARNINGS:")
#             for warning in self.warnings:
#                 report.append(f"  - {warning}")
                
#         if not self.errors and not self.warnings:
#             report.append("Energy conservation checks passed")
            
#         return "\n".join(report)
    
#     def clear(self):
#         """Clear diagnostics for next timestep"""
#         self.warnings = []
#         self.errors = []