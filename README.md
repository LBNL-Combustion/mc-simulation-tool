# flued_dynamics

Transient 1-D simulation of heat transfer, buoyancy-driven flow, and pressure in residential chimney/flue systems.

## What it does

Given a chimney geometry (pipe dimensions, materials, insulation, fittings) and boundary conditions (ambient temperatures, inlet gas temperature, door schedule, wind), the solver marches forward in time computing:

- Gas and wall temperatures along the flow path
- Mass flow rate (buoyancy-driven with friction, minor losses, and optional inlet restriction)
- Pressure distribution through the stack
- Condensation risk (dew point tracking)

The chimney is discretized into segments. Each timestep couples the momentum equation (quasi-steady with first-order lag) to the energy equation via Picard iteration.

## Quick start

```bash
pip install numpy scipy matplotlib
```

```python
from flued_module import Chimney

parts = [...]   # list of pipe section dicts (see sample_config.json)
elbows = [...]  # fittings between parts

ch = Chimney(
    parts=parts,
    elbows=elbows,
    dt=1.0,
    total_time=600,
    use_coupled_solver=True,
)
ch.run_simulation()

# Results: ch.T_gas, ch.T_wall, ch.velocity, ch.pressure, ch.mass_flow_rate
# All are numpy arrays, shape (time_steps+1, total_segments) or (time_steps+1,)
```

## CLI runner

`run_simulation.py` takes a JSON config and returns results as JSON. Useful for calling from a front end or other language.

```bash
python run_simulation.py sample_config.json --output results.json
```

See `sample_config.json` for the full schema with a 15ft chimney example including door schedule and inlet temperature ramp.

## Pipe types

**Single wall** -- bare steel pipe (indoor sections, typically 24-gauge black stovepipe):
```python
{'type': 'single', 'inner_diameter_ft': 0.5, 'height_ft': 6.125,
 'orientation': 'vertical', 'T_ambient_K': 295, 'Pa': 98457,
 'num_segments': 6, ...}
```

**Triple wall** -- insulated chimney pipe (liner / ceramic fiber / outer shell):
```python
{'type': 'triple', 'inner_diameter_ft': 0.5, 'height_ft': 5.917,
 'orientation': 'vertical', 'T_ambient_K': 275, 'Pa': 98457,
 'num_segments': 8,
 'k_inner_W_m_K': 26, 'k_middle_W_m_K': 0.04, 'k_outer_W_m_K': 16, ...}
```

`MaterialProperties` can build these dicts from material names:
```python
from flued_module.material_properties import MaterialProperties
mp = MaterialProperties('flued_module/data/pipe_props.json')
part = mp.create_triple_wall_chimney_part(
    inner_diameter_ft=0.5, height_ft=5.917,
    inner_material='stainless_430', middle_material='ceramic_fiber',
    outer_material='stainless_316', orientation='vertical',
    T_ambient_K=275, Pa=98457, num_segments=8)
```

## Fittings

Elbows and tees go in the `elbows` list. Each specifies which parts it connects and its geometry. Pressure loss uses K-factors; the thermal model optionally absorbs fitting length/mass into adjacent segments (`fittings_thermal` parameter).

## Inlet restriction and door schedule

The `inlet_restriction` dict models a stove door as a variable-area inlet. Pass a `door_opening_func` that returns percent open as a function of time:

```python
def door_func(t):
    if t < 90: return 100.0    # wide open
    if t < 300: return 11.7    # cracked to 2 inches
    return 0.0                 # closed (leakage only)

ch = Chimney(..., door_opening_func=door_func,
             inlet_restriction={'enable': True, 'A_leak_m2': 0.0125, ...})
```

## Solver parameters

The coupled solver uses Picard iteration with adaptive relaxation and a quasi-steady momentum model. Two parameters control how fast the flow responds to transients:

- `solver_relaxation` -- Picard under-relaxation factor (default 0.3)
- `solver_alpha_cap` -- max fraction of steady-state step per timestep (default 0.5)

```python
ch = Chimney(..., solver_relaxation=0.5, solver_alpha_cap=1.0)
```

## Module layout

```
flued_module/
    __init__.py
    chimney.py                  # Chimney class, simulation loop, pressure profiles
    coupled_solver.py           # CoupledFlowThermalSolver (momentum + energy coupling)
    physics.py                  # Air properties, Nu/Re correlations
    comprehensive_heat_transfer.py  # Wall heat transfer (convection + radiation)
    stable_triple_wall.py       # Triple-wall steady-state thermal resistance
    adaptive_triple_wall.py     # Triple-wall transient solver with inter-layer radiation
    elbows.py                   # Fitting K-factors and geometry
    condensation.py             # Dew point and condensation tracking
    material_properties.py      # Material database, part builders
    data/pipe_props.json        # Material property tables
run_simulation.py               # CLI/Python entry point (JSON in, JSON out)
sample_config.json              # Example configuration
```

## Requirements

- Python 3.8+
- numpy, scipy, matplotlib
