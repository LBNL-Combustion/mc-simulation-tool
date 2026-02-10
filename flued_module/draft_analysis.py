"""
Draft analysis and visualization for chimney simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .flow.elbows import base_K

@dataclass
class DraftResults:
    """Container for draft calculation results"""
    time: np.ndarray  # Time array (s)
    total_draft: np.ndarray  # Total draft pressure (Pa)
    segment_draft: np.ndarray  # Draft by segment (Pa)
    available_draft: np.ndarray  # Available draft after losses (Pa)
    draft_velocity: np.ndarray  # Theoretical velocity from draft (m/s)
    pressure_losses: np.ndarray  # Friction and fitting losses (Pa)
    mass_flow_rate: np.ndarray  # Actual mass flow (kg/s)
    stack_height: float  # Total chimney height (m)

class DraftAnalyzer:
    """Analyze and visualize chimney draft performance"""
    
    def __init__(self, chimney):
        """Initialize with chimney object"""
        self.chimney = chimney
        self.g = 9.81  # gravity (m/s²)
        
    def calculate_draft(self) -> DraftResults:
        """
        Calculate draft pressure for all timesteps
        
        Returns:
            DraftResults object with all draft calculations
        """
        n_steps = self.chimney.time_steps + 1
        n_segments = self.chimney.total_segments
        
        # Initialize arrays
        total_draft = np.zeros(n_steps)
        segment_draft = np.zeros((n_steps, n_segments))
        available_draft = np.zeros(n_steps)
        draft_velocity = np.zeros(n_steps)
        pressure_losses = np.zeros(n_steps)
        
        # Calculate total stack height
        stack_height = sum(part.height_m for part in self.chimney.parts)
        
        # For each timestep
        for step in range(n_steps):
            # Calculate segment-by-segment draft
            cumulative_draft = 0
            cumulative_loss = 0
            
            for seg in range(n_segments):
                part_idx, local_seg, part = self.chimney.get_segment_info(seg)
                segment_height = part.height_m / part.num_segments
                
                # Gas and ambient densities
                T_gas = self.chimney.T_gas[step, seg]
                T_ambient = part.T_ambient_K
                P_ambient = part.Pa
                
                rho_gas = self.chimney.density[step, seg]
                rho_ambient = self.chimney.physics.air_density(T_ambient, P_ambient)
                
                # Buoyancy pressure for this segment
                dp_buoyancy = self.g * segment_height * (rho_ambient - rho_gas)
                segment_draft[step, seg] = dp_buoyancy
                cumulative_draft += dp_buoyancy
                
                # Calculate pressure losses
                if step > 0:  # Skip first timestep (no flow yet)
                    velocity = self.chimney.velocity[step, seg]
                    diameter = part.inner_diameter_m
                    
                    # Friction loss
                    if abs(velocity) > 0.01:
                        Re = rho_gas * abs(velocity) * diameter / self.chimney.physics.air_viscosity(T_gas)
                        f = self._friction_factor(Re, part.roughness_m / diameter)
                        dp_friction = f * (segment_height / diameter) * 0.5 * rho_gas * velocity**2
                        
                        # Add fitting losses from actual elbow specs
                        elbow_key = (part_idx, local_seg)
                        if hasattr(self.chimney, 'segment_elbows') and elbow_key in self.chimney.segment_elbows:
                            for elbow_spec in self.chimney.segment_elbows[elbow_key]:
                                K_fitting = base_K(elbow_spec) * elbow_spec.vane_factor
                                dp_fitting = K_fitting * 0.5 * rho_gas * velocity**2
                                dp_friction += dp_fitting
                        
                        cumulative_loss += dp_friction
            
            total_draft[step] = cumulative_draft
            pressure_losses[step] = cumulative_loss
            available_draft[step] = cumulative_draft - cumulative_loss
            
            # Theoretical velocity from available draft (Bernoulli)
            if available_draft[step] > 0 and step > 0:
                avg_density = np.mean(self.chimney.density[step])
                draft_velocity[step] = np.sqrt(2 * available_draft[step] / avg_density)
        
        return DraftResults(
            time=self.chimney.time_array,
            total_draft=total_draft,
            segment_draft=segment_draft,
            available_draft=available_draft,
            draft_velocity=draft_velocity,
            pressure_losses=pressure_losses,
            mass_flow_rate=self.chimney.mass_flow_rate,
            stack_height=stack_height
        )
    
    def _friction_factor(self, Re: float, relative_roughness: float) -> float:
        """Calculate Darcy friction factor"""
        if Re < 2300:
            return 64 / max(Re, 1)
        elif relative_roughness < 1e-6:
            # Smooth pipe - Blasius
            return 0.316 / Re**0.25
        else:
            # Rough pipe - Colebrook approximation
            return 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re**0.9))**2
    
    def plot_draft_analysis(self, results: DraftResults, save_path: Optional[str] = None):
        """
        Create comprehensive draft analysis plots
        
        Args:
            results: DraftResults from calculate_draft()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Chimney Draft Analysis - Total Height: {results.stack_height:.1f}m', fontsize=14)
        
        # Plot 1: Total Draft vs Time
        ax1 = axes[0, 0]
        ax1.plot(results.time, results.total_draft, 'b-', linewidth=2, label='Total Draft')
        ax1.plot(results.time, results.available_draft, 'g--', linewidth=2, label='Available Draft')
        ax1.fill_between(results.time, results.available_draft, results.total_draft, 
                        alpha=0.3, color='red', label='Pressure Losses')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Pressure (Pa)')
        ax1.set_title('Draft Pressure vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Draft Components
        ax2 = axes[0, 1]
        # Show draft by height at final time
        final_step = -1
        heights = []
        drafts = []
        for seg in range(results.segment_draft.shape[1]):
            part_idx, local_seg, part = self.chimney.get_segment_info(seg)
            height = part.z_m + (local_seg + 0.5) * part.height_m / part.num_segments
            heights.append(height)
            drafts.append(results.segment_draft[final_step, seg])
        
        ax2.bar(range(len(heights)), drafts, color='blue', alpha=0.7)
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('Draft Contribution (Pa)')
        ax2.set_title(f'Draft by Segment at t={results.time[final_step]:.0f}s')
        ax2.grid(True, alpha=0.3)
        
        # Add height labels
        ax2_twin = ax2.twiny()
        ax2_twin.set_xlim(ax2.get_xlim())
        ax2_twin.set_xticks(range(len(heights)))
        ax2_twin.set_xticklabels([f'{h:.1f}m' for h in heights], rotation=45, fontsize=8)
        ax2_twin.set_xlabel('Height (m)')
        
        # Plot 3: Mass Flow and Velocity
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(results.time, results.mass_flow_rate, 'b-', linewidth=2, label='Mass Flow')
        line2 = ax3_twin.plot(results.time, results.draft_velocity, 'r--', linewidth=2, label='Draft Velocity')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Mass Flow Rate (kg/s)', color='b')
        ax3_twin.set_ylabel('Velocity (m/s)', color='r')
        ax3.set_title('Flow Rate and Velocity')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        ax3.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='best')
        
        # Plot 4: Efficiency and Performance
        ax4 = axes[1, 1]
        
        # Calculate efficiency (available/total draft)
        efficiency = np.zeros_like(results.total_draft)
        mask = results.total_draft > 1  # Avoid division by very small numbers
        efficiency[mask] = results.available_draft[mask] / results.total_draft[mask] * 100
        
        ax4.plot(results.time, efficiency, 'g-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Draft Efficiency (%)')
        ax4.set_title('Draft Efficiency (Available/Total)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # Add text box with key metrics
        steady_draft = results.total_draft[-1]
        steady_available = results.available_draft[-1]
        steady_flow = results.mass_flow_rate[-1]
        steady_velocity = results.draft_velocity[-1]
        
        textstr = f'Steady State Values:\n'
        textstr += f'Total Draft: {steady_draft:.1f} Pa\n'
        textstr += f'Available: {steady_available:.1f} Pa\n'
        textstr += f'Mass Flow: {steady_flow:.3f} kg/s\n'
        textstr += f'Velocity: {steady_velocity:.2f} m/s'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Draft analysis plot saved to: {save_path}")
        
        return fig
    
    def plot_draft_vs_temperature(self, results: DraftResults, save_path: Optional[str] = None):
        """
        Plot draft vs gas temperature relationship
        
        Args:
            results: DraftResults from calculate_draft()
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Draft vs Temperature Analysis', fontsize=14)
        
        # Calculate average gas temperature for each timestep
        avg_gas_temp = np.mean(self.chimney.T_gas - 273.15, axis=1)  # Convert to °C
        
        # Plot 1: Draft vs Average Gas Temperature
        ax1.scatter(avg_gas_temp, results.total_draft, alpha=0.6, s=20)
        ax1.set_xlabel('Average Gas Temperature (°C)')
        ax1.set_ylabel('Total Draft (Pa)')
        ax1.set_title('Draft vs Gas Temperature')
        ax1.grid(True, alpha=0.3)
        
        # Add theoretical line
        T_ambient = self.chimney.parts[0].T_ambient_K - 273.15
        T_range = np.linspace(T_ambient, avg_gas_temp.max(), 100)
        P_ambient = self.chimney.parts[0].Pa
        rho_ambient = self.chimney.physics.air_density(T_ambient + 273.15, P_ambient)
        theoretical_draft = []
        for T in T_range:
            rho_gas = self.chimney.physics.air_density(T + 273.15, P_ambient)
            dp = self.g * results.stack_height * (rho_ambient - rho_gas)
            theoretical_draft.append(dp)
        
        ax1.plot(T_range, theoretical_draft, 'r--', label='Theoretical (no losses)', linewidth=2)
        ax1.legend()
        
        # Plot 2: Temperature Profile at Steady State
        ax2.plot(self.chimney.T_gas[-1] - 273.15, label='Gas', marker='o')
        ax2.plot(self.chimney.T_wall[-1] - 273.15, label='Wall', marker='s')
        if hasattr(self.chimney, 'T_liner'):
            # Find triple wall segments
            for seg in range(self.chimney.total_segments):
                part_idx, _, part = self.chimney.get_segment_info(seg)
                if part.type == 'triple':
                    ax2.plot(seg, self.chimney.T_liner[-1, seg] - 273.15, 'g^', 
                            markersize=8, label='Liner' if seg == 0 else '')
                    ax2.plot(seg, self.chimney.T_middle[-1, seg] - 273.15, 'y*', 
                            markersize=10, label='Insulation' if seg == 0 else '')
                    ax2.plot(seg, self.chimney.T_outer[-1, seg] - 273.15, 'mv', 
                            markersize=8, label='Outer' if seg == 0 else '')
        
        ax2.set_xlabel('Segment Index')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature Profile (Steady State)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Draft vs temperature plot saved to: {save_path}")
        
        return fig
    
    def print_draft_summary(self, results: DraftResults):
        """Print summary of draft analysis results"""
        print("\n" + "="*60)
        print("CHIMNEY DRAFT ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nChimney Configuration:")
        print(f"  Total Height: {results.stack_height:.1f} m")
        print(f"  Number of segments: {self.chimney.total_segments}")
        
        print(f"\nSteady State Results (t={results.time[-1]:.0f}s):")
        print(f"  Total Draft Pressure: {results.total_draft[-1]:.1f} Pa")
        print(f"  Pressure Losses: {results.pressure_losses[-1]:.1f} Pa")
        print(f"  Available Draft: {results.available_draft[-1]:.1f} Pa")
        print(f"  Draft Efficiency: {results.available_draft[-1]/results.total_draft[-1]*100:.1f}%")
        
        print(f"\nFlow Performance:")
        print(f"  Mass Flow Rate: {results.mass_flow_rate[-1]:.4f} kg/s")
        print(f"  Average Velocity: {np.mean(self.chimney.velocity[-1]):.2f} m/s")
        print(f"  Theoretical Draft Velocity: {results.draft_velocity[-1]:.2f} m/s")
        
        print(f"\nTemperature Conditions:")
        avg_gas_temp = np.mean(self.chimney.T_gas[-1]) - 273.15
        inlet_temp = self.chimney.T_gas[-1, 0] - 273.15
        outlet_temp = self.chimney.T_gas[-1, -1] - 273.15
        print(f"  Inlet Gas Temperature: {inlet_temp:.1f}°C")
        print(f"  Outlet Gas Temperature: {outlet_temp:.1f}°C")
        print(f"  Average Gas Temperature: {avg_gas_temp:.1f}°C")
        print(f"  Temperature Drop: {inlet_temp - outlet_temp:.1f}°C")
        
        # Calculate required draft for different flow rates
        print(f"\nRequired Draft Analysis:")
        target_flows = [0.005, 0.01, 0.02, 0.05]  # kg/s
        avg_density = np.mean(self.chimney.density[-1])
        
        # Use actual loss ratio from simulation results
        if results.available_draft[-1] > 0:
            loss_ratio = results.total_draft[-1] / results.available_draft[-1]
        else:
            loss_ratio = 2.0  # Fallback if no available draft

        for mdot in target_flows:
            # Calculate required velocity
            avg_area = np.mean([part.inner_area_m2 for part in self.chimney.parts])
            v_required = mdot / (avg_density * avg_area)

            # Estimate required draft using actual loss ratio from simulation
            dp_required = 0.5 * avg_density * v_required**2 * loss_ratio
            
            print(f"  For {mdot:.3f} kg/s: Need ~{dp_required:.1f} Pa "
                  f"({'ACHIEVABLE' if dp_required < results.available_draft[-1] else 'NOT ACHIEVABLE'})")
        
        print("\n" + "="*60)