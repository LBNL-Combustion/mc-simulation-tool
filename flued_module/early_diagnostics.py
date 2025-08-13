"""
Early time diagnostics to catch unphysical behavior
"""
import numpy as np

class EarlyTimeDiagnostics:
    """
    Monitor early timesteps to catch instabilities before they grow
    """
    
    def __init__(self, ambient_temp_K=293.15):
        self.ambient_temp = ambient_temp_K
        self.errors = []
        self.warnings = []
        
    def check_timestep(self, step: int, time: float, T_gas: np.ndarray, 
                      T_wall: np.ndarray, T_liner: np.ndarray = None,
                      T_middle: np.ndarray = None, T_outer: np.ndarray = None,
                      mass_flow: float = 0, dt: float = 0):
        """
        Check for unphysical behavior at each timestep
        """
        # Clear previous diagnostics
        self.errors.clear()
        self.warnings.clear()
        
        # Check gas temperatures
        if np.any(T_gas < self.ambient_temp - 1.0):
            min_T = np.min(T_gas)
            seg = np.argmin(T_gas)
            self.errors.append(
                f"Gas temperature {min_T-273.15:.1f}°C below ambient at segment {seg}"
            )
            
        if np.any(T_gas < 0):
            self.errors.append("CRITICAL: Gas temperature below absolute zero!")
            
        # Check wall temperatures  
        if np.any(T_wall < self.ambient_temp - 1.0):
            min_T = np.min(T_wall)
            seg = np.argmin(T_wall)
            self.errors.append(
                f"Wall temperature {min_T-273.15:.1f}°C below ambient at segment {seg}"
            )
            
        if np.any(T_wall < 0):
            self.errors.append("CRITICAL: Wall temperature below absolute zero!")
            
        # Check triple wall if provided
        if T_liner is not None and np.any(T_liner < self.ambient_temp - 1.0):
            min_T = np.min(T_liner)
            self.errors.append(f"Liner temperature {min_T-273.15:.1f}°C below ambient")
            
        # Check for excessive cooling rates
        if step > 0 and dt > 0:
            # Maximum physically reasonable cooling rate
            max_cooling_rate = 100.0  # K/s
            
            # Check gas cooling rate (need previous values)
            # This would need to be implemented with history
            
        # Check mass flow
        if mass_flow < 0:
            self.errors.append(f"Negative mass flow: {mass_flow} kg/s")
            
        # Only print critical errors, not every warning
        if self.errors and "CRITICAL" in str(self.errors):
            print(f"\n*** CRITICAL ERROR at t={time:.1f}s (step {step}) ***")
            for error in self.errors:
                if "CRITICAL" in error:
                    print(f"ERROR: {error}")
                
    def get_report(self) -> str:
        """Get diagnostic report"""
        if not self.errors and not self.warnings:
            return "No issues detected"
            
        report = []
        if self.errors:
            report.append("ERRORS:")
            for e in self.errors:
                report.append(f"  {e}")
        if self.warnings:
            report.append("WARNINGS:")
            for w in self.warnings:
                report.append(f"  {w}")
        return "\n".join(report)