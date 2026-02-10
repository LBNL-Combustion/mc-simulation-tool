"""
Example usage of the PyFluesim package
Demonstrates the exact interface you requested

This script shows how to set up and run a basic chimney simulation
with a single-wall configuration. The simulation models heat transfer
and buoyancy-driven flow through the chimney system.
"""

import sys
import os
# Add the pyfluesim module to the Python path so we can import it
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyfluesim'))

from pyfluesim import Chimney

# STEP 1: DEFINE THE CHIMNEY GEOMETRY AND PROPERTIES
# Each part represents a section of the chimney with specific properties
# You can have multiple parts with different diameters, materials, etc.
parts = [
    {
        # WALL CONFIGURATION
        'type': 'single',           # 'single' for simple wall, 'triple' for insulated
        
        # GEOMETRY
        'inner_diameter_ft': 4/12,  # 4 inches converted to feet (this is the flow area)
        'height_ft': 7.5,           # Height of this chimney section
        'z_m': 0,                   # Elevation above reference point (for buoyancy calc)
        
        # WALL MATERIAL PROPERTIES (for single wall)
        'thickness_w_m': 0.00076,   # Wall thickness in meters (0.76mm steel)
        'density_w_kg_m3': 7850,    # Density of wall material (steel)
        'cp_w_J_kg_K': 460,         # Specific heat of wall material (steel)
        'k_w_W_m_K': 50,            # Thermal conductivity of wall (steel)
        'epsilon_w': 0.8,           # Emissivity of outer wall surface (for radiation)
        
        # OPERATING CONDITIONS
        'T_ambient_K': 294.15,      # Ambient temperature (21°C = 294.15K)
        'Pa': 101325,               # Atmospheric pressure in Pascals
        
        # ADVANCED SETTINGS
        'film_w': 5.08e-5,          # Max condensate film thickness in meters
        'orientation': 'vertical',   # 'vertical' or 'horizontal' affects convection
        
        # NUMERICAL SETTINGS
        'num_segments': 5,          # Number of computational cells for this part
                                   # More segments = better accuracy but slower
    }
]

# STEP 2: CREATE THE CHIMNEY SIMULATION OBJECT
# dt: time step in seconds (smaller = more accurate but slower)
# total_time: how long to simulate in seconds
# use_coupled_solver: True for better accuracy (iterative), False for faster (sequential)

# For FAST computation (3-5x faster), use:
# chimney = Chimney(parts=parts, dt=5.0, total_time=1200, use_coupled_solver=False)

# For NORMAL computation (balanced):
chimney = Chimney(parts=parts, dt=0.5, total_time=60, use_coupled_solver=True)

# Print initial configuration to help understand what we're simulating
print("Running chimney simulation...")
print(f"Configuration: {len(chimney.parts)} parts, {chimney.total_segments} total segments")
print(f"Time step: {chimney.dt}s, Total time: {chimney.total_time}s")

# STEP 3: RUN THE SIMULATION
# This is where all the physics happens - the simulation will:
# 1. Calculate inlet gas temperature over time (fire curve)
# 2. Solve heat transfer equations for each segment
# 3. Calculate buoyancy forces from density differences
# 4. Solve for mass flow rate and pressure distribution
# 5. Update temperatures based on energy balance
# save_interval: how often to save detailed results (1 = every timestep)
chimney.run_simulation(save_interval=1)

# STEP 4: EXTRACT AND DISPLAY RESULTS
# Get a summary of the key results
summary = chimney.get_summary_data()
print("\nSimulation Results:")
print(f"Final gas temperature range: {summary['final_gas_temp_range_C'][0]:.1f} to {summary['final_gas_temp_range_C'][1]:.1f} °C")
print(f"Final wall temperature range: {summary['final_wall_temp_range_C'][0]:.1f} to {summary['final_wall_temp_range_C'][1]:.1f} °C")
print(f"Final mass flow rate: {summary['final_mass_flow_rate_kg_s']:.4f} kg/s")

# STEP 5: VISUALIZE THE RESULTS
# Plot temperature profiles along the chimney height
print("\nGenerating temperature profile plots...")
chimney.plot_temperature_profiles()

print("\nSimulation completed successfully!")

# STEP 6: ACCESS RAW DATA FOR CUSTOM ANALYSIS
# All simulation data is stored in numpy arrays that you can access directly:
# - chimney.T_gas[time_step, segment]: Gas temperature in Kelvin
# - chimney.T_wall[time_step, segment]: Wall temperature in Kelvin  
# - chimney.mass_flow_rate[time_step]: Mass flow rate in kg/s
# - chimney.velocity[time_step, segment]: Gas velocity in m/s
print("\nDetailed results access:")
print(f"Gas temperatures at final time (°C): {chimney.T_gas[-1] - 273.15}")
print(f"Wall temperatures at final time (°C): {chimney.T_wall[-1] - 273.15}")
print(f"Mass flow rates over time (kg/s): min={chimney.mass_flow_rate.min():.4f}, max={chimney.mass_flow_rate.max():.4f}")