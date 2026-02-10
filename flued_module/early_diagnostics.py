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
                      mass_flow: float = 0, dt: float = 0,
                      elbow_losses: dict = None, friction_losses: dict = None):
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
            
        # Check mass flow - negative flow is valid (backdraft condition)
        # Only flag as info, not error, since backdraft simulation is intentional
        if mass_flow < 0:
            self.warnings.append(f"Backdraft detected: mass_flow = {mass_flow*1000:.2f} g/s")
            
        # Check elbow losses if provided
        if elbow_losses is not None:
            self._check_elbow_losses(elbow_losses, friction_losses)
            
        # Print immediate warnings for early timesteps
        if step < 10 and (self.errors or self.warnings):
            print(f"\n*** Diagnostics at t={time:.1f}s (step {step}) ***")
            for error in self.errors:
                print(f"ERROR: {error}")
            for warning in self.warnings:
                print(f"WARNING: {warning}")
                
    def _check_elbow_losses(self, elbow_losses: dict, friction_losses: dict = None):
        """
        Check elbow pressure losses for physical reasonableness
        
        Args:
            elbow_losses: Dict with 'K_total', 'Le_total', 'dp_minor' 
            friction_losses: Optional dict with 'dp_friction' for comparison
        """
        # Check K values
        if 'K_total' in elbow_losses:
            K_total = elbow_losses['K_total']
            if K_total < 0:
                self.errors.append(f"Negative total K value: {K_total}")
            elif K_total > 100:
                self.warnings.append(f"Unusually high total K value: {K_total}")
        
        # Check equivalent lengths
        if 'Le_total' in elbow_losses:
            Le_total = elbow_losses['Le_total']
            if Le_total < 0:
                self.errors.append(f"Negative total equivalent length: {Le_total}m")
            elif Le_total > 100:
                self.warnings.append(f"Unusually high equivalent length: {Le_total}m")
        
        # Check pressure drops
        if 'dp_minor' in elbow_losses:
            dp_minor = elbow_losses['dp_minor']
            if dp_minor < 0:
                self.errors.append(f"Negative elbow pressure drop: {dp_minor} Pa")
            elif dp_minor > 10000:  # 10 kPa is very high for elbow losses
                self.warnings.append(f"Very high elbow pressure drop: {dp_minor:.0f} Pa")
            
            # Compare with friction losses if available
            if friction_losses and 'dp_friction' in friction_losses:
                dp_friction = friction_losses['dp_friction']
                if dp_friction > 0:
                    ratio = dp_minor / dp_friction
                    if ratio > 20:
                        self.warnings.append(
                            f"Elbow losses dominate: {ratio:.1f}x friction losses"
                        )
                    elif ratio < 0.01 and dp_minor > 0:
                        self.warnings.append(
                            f"Elbow losses negligible: only {ratio*100:.1f}% of friction"
                        )
        
        # Check for NaN or Inf
        for key, value in elbow_losses.items():
            if np.isnan(value):
                self.errors.append(f"NaN detected in elbow {key}")
            elif np.isinf(value):
                self.errors.append(f"Inf detected in elbow {key}")
    
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