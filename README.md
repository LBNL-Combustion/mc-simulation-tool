# PyFluesim - Chimney Heat Transfer and Draft Simulation

A Python package for simulating heat transfer, mass flow, and pressure dynamics in chimney and flue systems. Converted from the original FLUESIM5.BAS program developed by Scanada Consultants Limited.

## Features

- **Heat Transfer Modeling**: Transient heat transfer in chimney walls and gas flow
- **Mass Flow Calculations**: Buoyancy-driven flow with pressure loss calculations
- **Segmented Analysis**: Divide chimney parts into multiple segments for detailed analysis
- **Visualization**: Temperature profile plots showing gas and wall temperatures
- **Flexible Configuration**: Support for multiple chimney parts with different properties

## Installation

1. **Install Dependencies**:
   ```bash
   pip install numpy matplotlib
   ```
   
   Or using the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add to Python Path** (if not installing as package):
   ```python
   import sys
   sys.path.append('path/to/pyfluesim')
   ```

## Usage

### Basic Example

```python
from pyfluesim import Chimney

# Define chimney configuration
parts = [
    {
        'type': 'single',
        'inner_diameter_ft': 4/12,  # 4 inches in feet
        'height_ft': 7.5,
        'z_m': 0,
        'thickness_w_m': 0.00076,
        'orientation': 'vertical',
        'density_w_kg_m3': 7850,  # steel
        'cp_w_J_kg_K': 460,       # steel specific heat
        'k_w_W_m_K': 50,          # steel thermal conductivity
        'film_w': 5.08e-5,
        'epsilon_w': 0.8,
        'T_ambient_K': 294.15,    # 21°C
        'Pa': 101325,
        'num_segments': 5,
    }
]

# Create and run simulation
chimney = Chimney(parts=parts, dt=0.5, total_time=60)
chimney.run_simulation(save_interval=1)

# View results
chimney.plot_temperature_profiles()
summary = chimney.get_summary_data()
print(f"Final temperature range: {summary['final_gas_temp_range_C']}")
```

### Configuration Parameters

Each part in the `parts` list should contain:

- **type**: 'single' (chimney type)
- **inner_diameter_ft**: Inner diameter in feet
- **height_ft**: Height in feet  
- **z_m**: Elevation above reference in meters
- **thickness_w_m**: Wall thickness in meters
- **orientation**: 'vertical' or 'horizontal'
- **density_w_kg_m3**: Wall material density (kg/m³)
- **cp_w_J_kg_K**: Wall material specific heat (J/kg·K)
- **k_w_W_m_K**: Wall thermal conductivity (W/m·K)
- **film_w**: Max condensate film thickness (m)
- **epsilon_w**: Wall emissivity (0-1)
- **T_ambient_K**: Ambient temperature (K)
- **Pa**: Atmospheric pressure (Pa)
- **num_segments**: Number of segments for this part

### Accessing Results

```python
# Temperature arrays (time_steps x segments)
gas_temps = chimney.T_gas      # Gas temperatures (K)
wall_temps = chimney.T_wall    # Inner wall temperatures (K)
outer_temps = chimney.T_outer  # Outer wall temperatures (K)

# Flow properties
mass_flows = chimney.mass_flow_rate  # Mass flow rate vs time (kg/s)
velocities = chimney.velocity        # Velocities (m/s)
pressures = chimney.pressure         # Pressures (Pa)

# Time array
times = chimney.time_array  # Time points (s)
```

## Package Structure

```
pyfluesim/
├── __init__.py          # Package initialization
├── chimney.py           # Main Chimney class
├── physics.py           # Physical properties and correlations
├── heat_transfer.py     # Heat transfer calculations
└── mass_flow.py         # Mass flow and pressure calculations
```

## Algorithm Background

This package implements algorithms from the original FLUESIM5.BAS program:

- **Heat Transfer**: Finite difference method for transient heat transfer
- **Mass Flow**: Buoyancy-driven flow with pressure network analysis
- **Physical Properties**: Temperature-dependent air properties
- **Segmentation**: Divides chimney into segments for spatial resolution

## Testing

Run the basic test to verify installation:

```bash
python test_basic.py
```

Run the full example:

```bash
python example_usage.py
```

## Original FLUESIM5 Reference

Based on FLUESIM5.BAS - Dynamic analysis program for furnace and flue transient performance, Version 5.0 (December 1989) developed by Scanada Consultants Limited for Canada Mortgage and Housing Corporation.

## Requirements

- Python 3.6+
- numpy >= 1.20.0
- matplotlib >= 3.3.0

## Notes

- The simulation uses explicit finite difference methods for temperature updates
- Mass flow calculations are simplified compared to the full BASIC version
- For production use, consider validating results against experimental data
- The original BASIC code contained extensive pressure network solving - this Python version implements a simplified approach suitable for single-chimney analysis