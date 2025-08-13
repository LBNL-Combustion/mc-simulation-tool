# Corrected Notebook Configuration

## Replace your notebook configuration with this:

```python
parts = [
    {
        # Bottom section - single wall (CORRECTED)
        'type': 'single',
        'inner_diameter_ft': 4/12,  # 4 inches converted to feet
        'height_ft': 7.5,
        'z_ft': 0,  # elevation above reference
        'thickness_w_m': 0.00076,  # wall thickness in meters
        'orientation': 'vertical',
        'density_w_kg_m3': 7850,  # density of wall material (steel)
        'cp_w_J_kg_K': 460,  # specific heat of wall material (steel)
        'k_w_W_m_K': 50,  # thermal conductivity of wall (steel)
        'film_w': 5.08e-5,  # max condensate film thickness in meters
        'epsilon_w': 0.8,  # emissivity of outer wall
        'T_ambient_K': 294.15,  # ambient temperature (21°C)
        'Pa': 101325,  # atmospheric pressure
        'num_segments': 5,
        # ADDED MISSING FIELDS:
        'has_fittings': False,  # augmentation for fittings
        'roughness_m': 1e-6,  # surface roughness
        'humidity_ratio': 0.01,  # kg water / kg dry air
    },
    {
        # Top section - triple wall 
        'type': 'triple',
        'inner_diameter_ft': 4/12,  # Same diameter as bottom section
        'height_ft': 5.0,  # Height of triple wall section
        'z_ft': 7.5,  # Starts where single wall ends
        'orientation': 'vertical',
        'T_ambient_K': 294.15,  # ambient temperature (21°C)
        'Pa': 101325,  # atmospheric pressure
        'num_segments': 4,  # num_segments_triple_layer

        # Inner wall (liner) properties - wall1
        'thickness_inner_m': 0.0005,  # triple_layer_T1_thickness_m
        'density_inner_kg_m3': 8000,  # wall1_density_kg_m3
        'cp_inner_J_kg_K': 500,       # wall1_cp_J_kg_K
        'k_inner_W_m_K': 16,          # wall1_k_W_m_K
        'epsilon_inner': 0.3,         # Lower emissivity for inner stainless steel

        # Middle layer (insulation) properties - wall2
        'thickness_middle_m': 0.0244,  # triple_layer_T2_thickness_m = 0.0254 - (0.0005 * 2)
        'density_middle_kg_m3': 150,   # wall2_density_kg_m3
        'cp_middle_J_kg_K': 1050,      # wall2_cp_J_kg_K
        'k_middle_W_m_K': 0.05,        # wall2_k_W_m_K

        # Outer wall properties - wall3
        'thickness_outer_m': 0.0005,   # triple_layer_T3_thickness_m
        'density_outer_kg_m3': 8000,   # wall3_density_kg_m3
        'cp_outer_J_kg_K': 500,        # wall3_cp_J_kg_K
        'k_outer_W_m_K': 16,           # wall3_k_W_m_K
        'epsilon_outer': 0.8,          # emissivity of outer wall

        # Air gaps (no gaps specified, so set to 0)
        'air_gap_inner_m': 0.0,        # No air gap between liner and insulation
        'air_gap_outer_m': 0.0,        # No air gap between insulation and outer wall

        # Additional properties
        'film_w': 5.08e-5,             # max condensate film thickness
        'has_fittings': False,         # augmentation for fittings
        'roughness_m': 1e-6,           # surface roughness
        'humidity_ratio': 0.01,        # kg water / kg dry air
    }
]

# Create chimney
chimney = Chimney(parts=parts, dt=0.5, total_time=600)
```

## Add this debug cell to verify the fix is loaded:

```python
# Verify the fix is loaded
import inspect
source = inspect.getsource(chimney.coupled_solver._solve_energy_improved)
if "CRITICAL FIX" in source:
    print("✅ Coupled solver has the wall temperature fix")
else:
    print("❌ Coupled solver does NOT have the fix - restart kernel!")
    
# Check initialization
print(f"\nInitialization check:")
for seg in range(chimney.total_segments):
    T_wall = chimney.T_wall[0, seg] - 273.15
    T_liner = chimney.T_liner[0, seg] - 273.15
    print(f"  Seg {seg}: T_wall[0,{seg}] = {T_wall:.1f}°C, T_liner[0,{seg}] = {T_liner:.1f}°C")
    if T_wall != 21.0:
        print(f"    ERROR: Should be 21.0°C!")
```

## Changes Made:

1. **Added missing fields** to single wall part:
   - `has_fittings`: False
   - `roughness_m`: 1e-6
   - `humidity_ratio`: 0.01

2. **All units verified** - everything looks correct

3. **Configuration matches** the working test cases