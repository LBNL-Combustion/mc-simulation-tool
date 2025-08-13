# FLUED_DYNAMICS - TECHNICAL API DOCUMENTATION

## PACKAGE OVERVIEW
The `flued_dynamics` package is a comprehensive Python simulation framework for modeling transient heat transfer and fluid flow in chimney/flue systems. It uses object-oriented design with modular components for physics calculations, numerical solvers, and diagnostics.

### Files Required for Simulation:
- **example_usage.ipynb** - Entry point showing how to use the package
- **flued_module/__init__.py** - Package initialization
- **flued_module/chimney.py** - Main chimney class
- **flued_module/physics.py** - Physical properties
- **flued_module/mass_flow.py** - Flow calculations
- **flued_module/comprehensive_heat_transfer.py** - Heat transfer solver
- **flued_module/coupled_solver.py** - Flow-thermal coupling
- **flued_module/condensation_physics.py** - Phase change physics
- **flued_module/stable_triple_wall.py** - Triple wall solver
- **flued_module/adaptive_triple_wall.py** - Adaptive solver
- **flued_module/energy_diagnostics.py** - Energy conservation
- **flued_module/early_diagnostics.py** - Startup checks
- **flued_module/draft_analysis.py** - Draft performance analysis
- **flued_module/chimney_geometry_plot.py** - Geometry visualization

### Files NOT Required:
- **time_simulation_plot.py** - Unused plotting module
- **vents_props.json** - Unused configuration file

## PACKAGE STRUCTURE
```
flued_dynamics/
├── __init__.py              # Package interface
├── example_usage.ipynb      # Usage examples and demos
└── flued_module/
    ├── __init__.py          # Module exports: Chimney, MassFlow, PhysicalProperties
    ├── chimney.py           # Main simulation controller
    ├── physics.py           # Thermophysical properties
    ├── mass_flow.py         # Flow and pressure calculations
    ├── comprehensive_heat_transfer.py  # Heat transfer solver
    ├── coupled_solver.py    # Flow-thermal coupling
    ├── condensation_physics.py  # Phase change modeling
    ├── stable_triple_wall.py    # Implicit wall solver
    ├── adaptive_triple_wall.py  # Adaptive wall solver
    ├── energy_diagnostics.py    # Conservation checks
    └── early_diagnostics.py     # Startup diagnostics
```

---

## MODULE: chimney.py - Chimney structure definition
The simulation begins by defining the chimney system as a series of connected parts

### CLASS: ChimneyPart (dataclass)
**Purpose**: Configuration container for a single chimney section
- Stores all properties for one chimney section (single or triple wall)
- Converts imperial units to SI units automatically for calculations
- Validates inputs and calculates derived geometric properties
- Supports two wall types: 'single' (single-wall) or 'triple' (class A)

#### Current Attributes (subject to change):

##### Required Parameters:
- **type** (str): Determines wall construction type
  - Values: 'single' | 'triple'
  
- **inner_diameter_ft** (float): Inner diameter in feet (converted to meters internally)
  - Range: > 0
  
- **height_ft** (float): Section height in feet (converted to meters internally)
  - Range: > 0
  
- **z_ft** (float): Elevation above reference point in feet (converted to meters)
  - Range: >= 0
  
- **orientation** (str): Orientation of pipe, affects natural convection correlations
  - Values: 'vertical' | 'horizontal'
  
- **T_ambient_K** (float): Ambient temperature of part in Kelvin
  - Range: not implemented but typically 273-320K
  
- **Pa** (float): Atmospheric pressure in Pascals
  - Default: 101325 Pa
  
- **num_segments** (int): Number of computational cells for this part
  - Range: >= 1

##### Single Wall Parameters (used when type='single')
The following parameters are currently defined by user, but in the future some will be mapped from a JSON given wall material/s, and thickness/es

- **thickness_w_m** (float): Wall thickness in meters
  - Default: 0.001, Range: > 0
  
- **density_w_kg_m3** (float): Wall material density in kg/m³
  - Default: 7850 (steel), Typical: 7850 (steel), 2700 (aluminum), others
  
- **cp_w_J_kg_K** (float): Wall specific heat in J/(kg·K)
  - Default: 460, Typical: 460 (steel), 900 (aluminum), others
  
- **k_w_W_m_K** (float): Wall thermal conductivity in W/(m·K)  
  - Default: 50 (steel)
  
- **epsilon_w** (float): Wall emissivity for radiation
  - Default: 0.8, Range: 0-1 (0=perfect reflector, 1=black body)

##### Triple Wall Parameters (used when type='triple')
Each of inner wall, middle layer, and outer layer must have:

- **Thicknesses** (floats): 
  - thickness_liner_m, thickness_insulation_m, thickness_outer_m
  - Thicknesses of layers in meters
  
- **Densities** (floats):
  - density_liner_kg_m3, density_insulation_kg_m3, density_outer_kg_m3
  
- **Specific heats** (floats):
  - cp_liner_J_kg_K, cp_insulation_J_kg_K, cp_outer_J_kg_K
  
- **Thermal conductivities** (floats):
  - k_liner_W_m_K, k_insulation_W_m_K, k_outer_W_m_K

In addition, all three layers must have:
- **Emissivities** (floats):
  - epsilon_liner, epsilon_insulation, epsilon_outer

##### Other General Parameters:
- **water_vapor_fraction** (float): Initial water vapor mole fraction
  - Default: 0.01
  
- **roughness** (float): Surface roughness in meters (affects friction factor)
  - Default: 0.0

##### Computed Attributes (set in __post_init__):
- **inner_diameter**: float - Converted from feet
- **height**: float - Converted from feet  
- **z**: float - Converted from feet
- **inner_area**: float - Cross-sectional area (π*d²/4)
- **inner_perimeter**: float - Wetted perimeter (π*d)
- **outer_diameter**: float - Calculated based on wall thickness

For triple wall only:
- **liner_outer_diameter**: float
- **insulation_inner_diameter**: float
- **insulation_outer_diameter**: float
- **outer_inner_diameter**: float

#### Methods:
- **__post_init__()**: Validates inputs and calculates derived properties

### CLASS: Chimney
**Purpose**: Main simulation orchestrator that manages the entire system
- Creates and validates ChimneyPart objects from configuration dictionaries
- Divides each part into computational segments for numerical solution
- Initializes physics modules (PhysicalProperties, ComprehensiveHeatTransfer, MassFlow, etc.)
- Manages time stepping and coordinates all physics calculations
- Stores complete history of temperatures, flows, and pressures

#### Constructor:
```python
__init__(parts: List[Union[ChimneyPart, Dict]], dt: float = 0.5, total_time: float = 60.0, 
         use_coupled_solver: bool = True)
```

##### Parameters:
- **parts**: List[Union[ChimneyPart, Dict]]
  - List of ChimneyPart objects or dictionaries with ChimneyPart parameters
  - Each dict must contain required ChimneyPart fields
  
- **dt**: float = 0.5
  - Simulation timestep in seconds
  - Smaller values increase accuracy but slow computation
  - For triple walls: recommend dt < 1s
  
- **total_time**: float = 60.0
  - Total simulation duration in seconds
  
- **use_coupled_solver**: bool = True
  - If True: Use iterative coupling between flow and thermal equations
  - If False: Use faster sequential solution
  - Coupled is more accurate but ~2-3x slower

#### Attributes:

##### Configuration:
- **parts**: List[ChimneyPart] - Validated chimney sections
- **dt**: float - Timestep size
- **total_time**: float - Simulation duration  
- **time_steps**: int - Number of timesteps
- **total_segments**: int - Sum of all part segments

##### Physics Modules:
- **physics**: PhysicalProperties - Gas property calculator
- **mass_flow_calc**: MassFlow - Flow calculations
- **comprehensive_ht**: ComprehensiveHeatTransfer - Advanced heat transfer
- **coupled_solver**: CoupledFlowThermalSolver - Iterative flow-thermal coupling
- **condensation**: CondensationPhysics - Condensation modeling
- **stable_solver**: StableTripleWallSolver - Implicit triple wall solver
- **adaptive_solver**: AdaptiveTripleWallSolver - Adaptive triple wall solver

##### State Arrays [time_steps+1, total_segments]:
- **T_gas**: np.ndarray - Gas temperature (K)
- **T_wall**: np.ndarray - Wall temperature (K) 
- **T_outer**: np.ndarray - Outer surface temperature (K)
- **T_liner**: np.ndarray - Triple wall inner liner (K)
- **T_insulation**: np.ndarray - Triple wall middle layer (K)
- **velocity**: np.ndarray - Gas velocity (m/s)
- **density**: np.ndarray - Gas density (kg/m³)
- **heat_flux_wall**: np.ndarray - Wall heat flux (W/m²)
- **h_conv_inner**: np.ndarray - Inner convection coefficient (W/m²K)
- **h_conv_outer**: np.ndarray - Outer convection coefficient (W/m²K)
- **pressure_drop**: np.ndarray - Segment pressure drop (Pa)
- **draft_pressure**: np.ndarray - Cumulative draft pressure (Pa)
- **water_vapor_mass_fraction**: np.ndarray - Water vapor mass fraction
- **condensation_rate**: np.ndarray - Condensation rate (kg/s)
- **dew_point_temperature**: np.ndarray - Dew point temperature (K)

##### Scalar Arrays [time_steps+1]:
- **mass_flow_rate**: np.ndarray - Mass flow rate (kg/s) 
- **time_array**: np.ndarray - Time points (s)

##### Geometry Arrays [total_segments]:
- **segment_heights**: np.ndarray - Height of each segment center (m)
- **segment_lengths**: np.ndarray - Length of each segment (m)
- **segment_areas**: np.ndarray - Inner surface area per segment (m²)
- **segment_volumes**: np.ndarray - Gas volume per segment (m³)

#### Methods:

##### run_simulation(save_interval: int = 10) -> None
**Purpose**: Execute the main simulation loop

**Parameters**:
- **save_interval**: int = 10 - Print progress every N timesteps (0 = no printing)

**Side Effects**:
- Fills all state arrays with simulation results
- Prints progress and diagnostics  
- Updates temperatures, flow rates, pressures

**Process**:
1. Main time loop from step 1 to time_steps:
   - Update current time
   - Calculate inlet temperature using _default_inlet_temp()
   - Solve physics equations (coupled or sequential)
   - Run diagnostics if enabled
   - Print progress periodically
3. If using coupled solver:
   - Call coupled_solver.solve_coupled_timestep()
   - Track convergence metrics
4. If using sequential solver:
   - Call _solve_comprehensive_heat_transfer()
   - Call _update_flow_and_pressure()
5. Run diagnostics and print progress

##### _default_inlet_temp(t: float, T_ambient: float = 293.15, T_max: float = 773.15, tau: float = 300) -> float
**Purpose**: Calculate inlet temperature with exponential ramp-up
**Parameters**:
- **t**: float - Time in seconds
- **T_ambient**: float = 293.15 - Starting ambient temperature (K)
- **T_max**: float = 773.15 - Maximum temperature (K) 
- **tau**: float = 300 - Time constant (s), temperature reaches 63% of max at t=tau
**Returns**: float - Inlet temperature (K)
**Formula**: T = T_ambient + (T_max - T_ambient) * (1 - exp(-t/tau))

##### _default_inlet_temp_fire(t_sec: float) -> float
**Purpose**: Calculate inlet temperature using ISO 834 fire curve

**Parameters**:
- **t_sec**: float - Time in seconds since ignition

**Returns**: float - Temperature in Kelvin

**Notes**: 
- Uses ISO 834: T = 20 + 345*log10(8t + 1) where t is in minutes
- Scales to 50% for flue gases (empirical)
- Caps at 800K for safety

##### _solve_comprehensive_heat_transfer(step: int, T_inlet: float) -> None
**Purpose**: Solve heat transfer for all segments at current time step

**Parameters**:
- **step**: int - Current time step index
- **T_inlet**: float - Inlet gas temperature (K)

**Process**:
1. Initialize with inlet conditions
2. For each segment (bottom to top):
   - Get segment properties and geometry
   - Call comprehensive heat transfer solver
   - Update gas and wall temperatures
   - Calculate condensation if present
   - Set outlet as next segment's inlet

##### _update_flow_and_pressure(step: int) -> None
**Purpose**: Update mass flow rate and pressure distribution

**Parameters**:
- **step**: int - Current time step index

**Process**:
1. Calculate total buoyancy pressure
2. Calculate total flow resistance
3. Solve for mass flow rate
4. Update velocities in each segment
5. Calculate pressure drops

##### get_summary_data() -> Dict[str, Any]
**Purpose**: Generate summary statistics and final results

**Returns**: Dictionary containing:
- total_segments: Total number of segments
- total_time: Simulation duration (s)
- time_steps: Number of time steps
- final_gas_temp_range_C: Tuple of (min, max) gas temps in °C
- final_wall_temp_range_C: Tuple of (min, max) wall temps in °C
- final_mass_flow_rate_kg_s: Final mass flow rate (kg/s)
- parts_config: List of part configurations with type, height, diameter, segments

##### check_conservation() -> None
**Purpose**: Check mass and energy conservation
**Side Effects**: Prints detailed conservation diagnostics

##### _check_energy_conservation() -> Dict[str, float]
**Purpose**: Check energy conservation at final time

**Returns**: Dictionary containing:
- energy_in_total: Total energy input (J)
- energy_stored_gas: Energy stored in gas (J)
- energy_stored_walls: Energy stored in walls (J)
- energy_lost_total: Energy lost to ambient (J)
- energy_balance: Input - stored - lost (J)
- percent_error: 100 * balance / input

**Note**: Called internally by check_conservation()

##### _check_mass_conservation() -> Dict[str, float]
**Purpose**: Check mass conservation through the chimney
**Returns**: Dictionary with mass flow statistics
**Note**: Called internally by check_conservation()

##### _initialize_arrays() -> None
**Purpose**: Pre-allocate all numpy arrays for simulation
**Side Effects**: Creates all state and geometry arrays

##### _set_initial_conditions() -> None
**Purpose**: Set initial temperatures and flow conditions
**Side Effects**: Fills arrays at time index 0



------
#### The following are for plotting/checking results and inputs
##### plot_temperature_profiles(times_to_plot: Optional[List[float]] = None) -> None
**Purpose**: Plot temperature vs height at specified times

**Parameters**:
- **times_to_plot**: Optional[List[float]] - Times in seconds to plot
  - If None, plots at t=0, 25%, 50%, 75%, 100% of total time

**Side Effects**: 
- Creates matplotlib figure showing gas and wall temperatures vs height


##### analyze_draft(plot: bool = True, save_plots: bool = False, plot_dir: str = './') -> Dict
**Purpose**: Analyze chimney draft performance
**Parameters**:
- **plot**: bool = True - Whether to create plots
- **save_plots**: bool = False - Whether to save plots to disk
- **plot_dir**: str = './' - Directory for saving plots
**Returns**: Dictionary with draft analysis results

##### plot_segment_diagnostics() -> None
**Purpose**: Create detailed diagnostic plots for each segment
**Side Effects**: Creates multi-panel figure with temperatures, velocities, heat transfer coefficients

##### get_segment_info(segment_idx: int) -> tuple
**Purpose**: Get part index and local segment index for a global segment
**Parameters**:
- **segment_idx**: int - Global segment index (0 to total_segments-1)
**Returns**: Tuple of (part_idx, local_segment_idx)

##### temp_evolution_by_part(ambient_temp: bool = False) -> None
**Purpose**: Plot temperature evolution over time for each chimney part
**Parameters**:
- **ambient_temp**: bool = False - Whether to show ambient temperature line
**Side Effects**: Creates figure with subplots for each part

##### temp_evolution_br() -> None
**Purpose**: Plot temperature evolution at bottom and top of chimney
**Side Effects**: Creates figure showing inlet/outlet temperatures over time

##### plot_draft() -> None
**Purpose**: Plot draft pressure distribution along chimney height
**Side Effects**: Creates figure showing pressure vs height at different times

##### plot_buoyancy_pressure() -> None
**Purpose**: Plot buoyancy pressure distribution
**Side Effects**: Creates figure showing buoyancy driving pressure vs height

##### temp_evolution_by_segment() -> None
**Purpose**: Plot temperature evolution for each segment (creates many subplots)
**Side Effects**: Creates large figure with subplot for each segment

##### _plot_heat_transfer_coefficients() -> None
**Purpose**: Plot heat transfer coefficients along chimney
**Side Effects**: Creates figure showing h_inner and h_outer vs height

##### _print_detailed_segment_info() -> None
**Purpose**: Print detailed information about each segment
**Side Effects**: Prints table with geometry and properties

##### plot_chimney_snapshot(timestep: int = -1, figsize: tuple = (10, 8), wall_scale: float = 5.0, show_temps: bool = True, show_grid: bool = True, save_fig: bool = False, filename: str = 'chimney_snapshot.png') -> None
**Purpose**: Create detailed snapshot of chimney state at specific time
**Parameters**:
- **timestep**: int = -1 - Time index (-1 for last timestep)
- **figsize**: tuple = (10, 8) - Figure size in inches
- **wall_scale**: float = 5.0 - Exaggeration factor for wall thickness
- **show_temps**: bool = True - Whether to show temperature annotations
- **show_grid**: bool = True - Whether to show grid lines
- **save_fig**: bool = False - Whether to save figure
- **filename**: str = 'chimney_snapshot.png' - Filename if saving

---

## MODULE: physics.py - Thermophysical Properties
Provides temperature-dependent fluid properties and dimensionless numbers

### CLASS: PhysicalProperties
**Purpose**: Calculate air/gas properties at various temperatures

#### Constants:
- **GRAVITY**: float = 9.81 - Gravitational acceleration (m/s²)
- **GAS_CONSTANT_AIR**: float = 287.05 - Specific gas constant for air (J/kg·K)

#### Methods:

##### air_density(temperature: float, pressure: float = 101325) -> float
**Purpose**: Calculate air density using ideal gas law

**Parameters**:
- **temperature**: float - Temperature in Kelvin
- **pressure**: float = 101325 - Pressure in Pascals

**Returns**: float - Density in kg/m³

**Formula**: ρ = P/(R*T)

##### air_viscosity(temperature: float) -> float  
**Purpose**: Calculate dynamic viscosity using Sutherland's law

**Parameters**:
- **temperature**: float - Temperature in Kelvin

**Returns**: float - Dynamic viscosity in Pa·s

**Formula**: μ = μ₀*(T/T₀)^(3/2)*(T₀+S)/(T+S)
- μ₀ = 1.716e-5 Pa·s at T₀ = 273.15 K
- S = 110.4 K (Sutherland constant)

##### air_thermal_conductivity(temperature: float) -> float
**Purpose**: Calculate thermal conductivity correlation

**Parameters**:
- **temperature**: float - Temperature in Kelvin

**Returns**: float - Thermal conductivity in W/(m·K)

**Formula**: k = 0.0241*(T/273.15)^0.9

##### air_specific_heat(temperature: float) -> float
**Purpose**: Calculate specific heat at constant pressure

**Parameters**:
- **temperature**: float - Temperature in Kelvin

**Returns**: float - Specific heat in J/(kg·K)

**Formula**: cp = 1000 + 0.17*(T - 273.15)

##### nusselt_natural_convection_vertical_cylinder(rayleigh: float, height_diameter_ratio: float) -> float
**Purpose**: Calculate Nusselt number for natural convection on vertical cylinder

**Parameters**:
- **rayleigh**: float - Rayleigh number (dimensionless)
- **height_diameter_ratio**: float - H/D ratio

**Returns**: float - Nusselt number (dimensionless)

**Correlations**:
- Ra < 10⁹: Nu = 0.68 + 0.67*Ra^0.25/[1 + (0.492/Pr)^(9/16)]^(4/9)
- Ra > 10⁹: Nu = {0.825 + 0.387*Ra^(1/6)/[1 + (0.492/Pr)^(9/16)]^(8/27)}²

##### friction_factor_smooth_tube(reynolds: float) -> float
**Purpose**: Calculate Darcy friction factor for smooth pipes

**Parameters**:
- **reynolds**: float - Reynolds number

**Returns**: float - Darcy friction factor (dimensionless)

**Correlations**:
- Re < 2300 (laminar): f = 64/Re
- Re >= 2300 (turbulent): f = 0.316/Re^0.25

---

## MODULE: mass_flow.py - Flow and Pressure Calculations
Calculates pressure drops, flow rates, and system resistance

### CLASS: MassFlow
**Purpose**: Handle all flow-related calculations for chimney system

#### Constructor:
```python
__init__(physics: Optional[PhysicalProperties] = None)
```

**Parameters**:
- **physics**: Optional[PhysicalProperties] - Physics property calculator
  - If None, creates new instance

#### Methods:

##### buoyancy_pressure(height: float, temp_gas: float, temp_ambient: float, pressure: float) -> float
**Purpose**: Calculate buoyancy driving pressure (stack effect)

**Parameters**:
- **height**: float - Vertical height (m)
- **temp_gas**: float - Gas temperature (K)
- **temp_ambient**: float - Ambient temperature (K)
- **pressure**: float - Atmospheric pressure (Pa)

**Returns**: float - Buoyancy pressure difference (Pa)

**Formula**: ΔP = g*h*(ρ_ambient - ρ_gas)

##### pressure_loss_friction(velocity: float, diameter: float, length: float, density: float, viscosity: float, roughness: float = 0.0) -> float
**Purpose**: Calculate frictional pressure loss using Darcy-Weisbach

**Parameters**:
- **velocity**: float - Flow velocity (m/s)
- **diameter**: float - Pipe diameter (m)
- **length**: float - Pipe length (m)
- **density**: float - Fluid density (kg/m³)
- **viscosity**: float - Dynamic viscosity (Pa·s)
- **roughness**: float = 0.0 - Surface roughness (m)

**Returns**: float - Pressure loss (Pa)

**Formula**: ΔP = f*(L/D)*(ρv²/2)

**Notes**:
- Uses Colebrook-White for rough pipes
- Falls back to smooth pipe if convergence fails

##### pressure_loss_minor(velocity: float, density: float, k_factor: float = 1.0) -> float
**Purpose**: Calculate minor losses (entrances, exits, fittings)

**Parameters**:
- **velocity**: float - Flow velocity (m/s)
- **density**: float - Fluid density (kg/m³)
- **k_factor**: float = 1.0 - Loss coefficient

**Returns**: float - Pressure loss (Pa)

**Formula**: ΔP = K*(ρv²/2)

##### calculate_mass_flow_rate(pressure_difference: float, system_resistance: float, density: float, area: float) -> float
**Purpose**: Calculate mass flow rate from pressure balance

**Parameters**:
- **pressure_difference**: float - Net driving pressure (Pa)
- **system_resistance**: float - Total flow resistance (Pa·s²/kg²)
- **density**: float - Fluid density (kg/m³)
- **area**: float - Cross-sectional area (m²)

**Returns**: float - Mass flow rate (kg/s)

**Notes**:
- Iteratively solves momentum equation
- Includes damping for stability

##### system_resistance_analysis(segments_data: List[Dict]) -> Dict[str, Any]
**Purpose**: Analyze system flow resistance components

**Parameters**:
- **segments_data**: List[Dict] - List of segment data dictionaries

**Returns**: Dictionary containing:
- total_friction_loss: Total friction pressure loss (Pa)
- total_minor_loss: Total minor pressure loss (Pa)
- total_buoyancy: Total buoyancy pressure (Pa)
- net_driving_pressure: Net pressure difference (Pa)
- individual_losses: List of losses per segment

##### entrance_loss_coefficient(geometry: str = 'sharp') -> float
**Purpose**: Get entrance loss coefficient

**Parameters**:
- **geometry**: str = 'sharp' - Type of entrance ('sharp', 'rounded', 'bellmouth')

**Returns**: float - Loss coefficient K

##### exit_loss_coefficient() -> float
**Purpose**: Get exit loss coefficient
**Returns**: float - Loss coefficient K = 1.0

##### elbow_loss_coefficient(angle_degrees: float, diameter: float, radius_curvature: Optional[float] = None) -> float
**Purpose**: Calculate elbow/bend loss coefficient

**Parameters**:
- **angle_degrees**: float - Bend angle in degrees
- **diameter**: float - Pipe diameter (m)
- **radius_curvature**: Optional[float] - Radius of curvature (m)

**Returns**: float - Loss coefficient K

##### draft_calculation(chimney_height: float, T_gas_avg: float, T_ambient: float, pressure_ambient: float) -> Dict[str, float]
**Purpose**: Calculate theoretical chimney draft

**Parameters**:
- **chimney_height**: float - Total chimney height (m)
- **T_gas_avg**: float - Average gas temperature (K)
- **T_ambient**: float - Ambient temperature (K)
- **pressure_ambient**: float - Ambient pressure (Pa)

**Returns**: Dictionary containing:
- draft_pressure: Theoretical draft (Pa)
- gas_density: Average gas density (kg/m³)
- ambient_density: Ambient air density (kg/m³)

##### wind_pressure_effect(wind_speed: float, wind_direction_deg: float, chimney_height: float, chimney_diameter: float, terrain_type: str = 'suburban') -> Dict[str, float]
**Purpose**: Calculate wind effects on chimney draft

**Parameters**:
- **wind_speed**: float - Wind speed at 10m height (m/s)
- **wind_direction_deg**: float - Wind direction (degrees)
- **chimney_height**: float - Chimney height (m)
- **chimney_diameter**: float - Chimney diameter (m)
- **terrain_type**: str = 'suburban' - Terrain category

**Returns**: Dictionary with wind pressure effects

##### flow_network_solver(nodes: List[Dict], connections: List[Dict], boundary_conditions: Dict, max_iterations: int = 50, tolerance: float = 1e-6) -> Dict[str, Any]
**Purpose**: Solve flow network using Hardy Cross method

**Parameters**:
- **nodes**: List[Dict] - Network nodes
- **connections**: List[Dict] - Pipe connections
- **boundary_conditions**: Dict - Pressure/flow boundaries
- **max_iterations**: int = 50 - Maximum iterations
- **tolerance**: float = 1e-6 - Convergence tolerance

**Returns**: Dictionary with node pressures and connection flows

##### equivalent_leakage_area(flow_rate: float, pressure_difference: float, reference_pressure: float = 10.0, exponent: float = 0.5) -> float
**Purpose**: Calculate equivalent leakage area

**Parameters**:
- **flow_rate**: float - Volumetric flow rate (m³/s)
- **pressure_difference**: float - Pressure difference (Pa)
- **reference_pressure**: float = 10.0 - Reference pressure (Pa)
- **exponent**: float = 0.5 - Flow exponent

**Returns**: float - Equivalent leakage area (m²)

---

## MODULE: comprehensive_heat_transfer.py - Heat Transfer Solver
Comprehensive heat transfer calculations including natural, forced, and mixed convection, radiation, and condensation

### CLASS: FlowProperties (dataclass)
**Purpose**: Container for fluid properties at a given state

#### Attributes:
- **temperature**: float - Temperature (K)
- **density**: float - Density (kg/m³)
- **viscosity**: float - Dynamic viscosity (Pa·s)
- **thermal_conductivity**: float - Thermal conductivity (W/m·K)
- **specific_heat**: float - Specific heat (J/kg·K)

### CLASS: ComprehensiveHeatTransfer
**Purpose**: Advanced heat transfer solver with all modes

#### Constructor:
```python
__init__(physics: Optional[PhysicalProperties] = None, 
         condensation: Optional[CondensationPhysics] = None)
```

**Parameters**:
- **physics**: Optional[PhysicalProperties] - Property calculator
- **condensation**: Optional[CondensationPhysics] - Condensation model

#### Methods:

##### gas_properties(temperature: float, pressure: float, water_vapor_fraction: float = 0.0) -> FlowProperties
**Purpose**: Calculate gas mixture properties including humidity effects

**Parameters**:
- **temperature**: float - Temperature (K)
- **pressure**: float - Pressure (Pa)
- **water_vapor_fraction**: float = 0.0 - Mole fraction of water vapor

**Returns**: FlowProperties object with mixture properties

**Notes**:
- Accounts for water vapor in all properties
- Uses mole-fraction weighting for mixture

##### reynolds_number(velocity: float, diameter: float, props: FlowProperties) -> float
**Purpose**: Calculate Reynolds number with bounds checking

**Parameters**:
- **velocity**: float - Flow velocity (m/s)
- **diameter**: float - Characteristic length (m)
- **props**: FlowProperties - Fluid properties

**Returns**: float - Reynolds number (dimensionless)

**Formula**: Re = ρ*v*D/μ

**Notes**:
- Bounds: 1.0 <= Re <= 1e8
- Protects against zero velocity/viscosity

##### prandtl_number(props: FlowProperties) -> float
**Purpose**: Calculate Prandtl number

**Parameters**:
- **props**: FlowProperties - Fluid properties

**Returns**: float - Prandtl number (dimensionless)

**Formula**: Pr = μ*cp/k

##### grashof_number(diameter: float, T_wall: float, T_bulk: float, props: FlowProperties) -> float
**Purpose**: Calculate Grashof number for natural convection

**Parameters**:
- **diameter**: float - Characteristic length (m)
- **T_wall**: float - Wall temperature (K)
- **T_bulk**: float - Bulk fluid temperature (K)
- **props**: FlowProperties - Fluid properties

**Returns**: float - Grashof number (dimensionless)

**Formula**: Gr = g*β*ΔT*L³*ρ²/μ²

##### rayleigh_number(grashof: float, prandtl: float) -> float
**Purpose**: Calculate Rayleigh number

**Parameters**:
- **grashof**: float - Grashof number
- **prandtl**: float - Prandtl number

**Returns**: float - Rayleigh number (dimensionless)

**Formula**: Ra = Gr*Pr

##### nusselt_forced_convection_internal(reynolds: float, prandtl: float, length_diameter_ratio: Optional[float] = None) -> float
**Purpose**: Calculate forced convection Nusselt number for internal flow

**Parameters**:
- **reynolds**: float - Reynolds number
- **prandtl**: float - Prandtl number
- **length_diameter_ratio**: Optional[float] - L/D for entrance effects

**Returns**: float - Nusselt number (dimensionless)

**Correlations**:
- Re < 2300 (laminar): Nu = 3.66 or entrance effects
- 2300 <= Re < 4000 (transition): Linear interpolation
- Re >= 4000 (turbulent): Nu = 0.023*Re^0.8*Pr^0.4 (Dittus-Boelter)

##### nusselt_natural_convection_external(rayleigh: float, orientation: str, height_diameter_ratio: float = 1.0) -> float
**Purpose**: Calculate natural convection Nusselt number for external surfaces

**Parameters**:
- **rayleigh**: float - Rayleigh number
- **orientation**: str - 'vertical' or 'horizontal'
- **height_diameter_ratio**: float = 1.0 - H/D ratio

**Returns**: float - Nusselt number (dimensionless)

**Notes**: Uses Churchill-Chu correlations

##### nusselt_natural_convection_internal(rayleigh: float, orientation: str, length_diameter_ratio: float = 10.0) -> float
**Purpose**: Calculate natural convection Nusselt number for internal flow

**Parameters**:
- **rayleigh**: float - Rayleigh number
- **orientation**: str - 'vertical' or 'horizontal'
- **length_diameter_ratio**: float = 10.0 - L/D ratio

**Returns**: float - Nusselt number (dimensionless)

##### combined_convection_coefficient(h_forced: float, h_natural: float) -> float
**Purpose**: Combine forced and natural convection coefficients

**Parameters**:
- **h_forced**: float - Forced convection coefficient (W/m²·K)
- **h_natural**: float - Natural convection coefficient (W/m²·K)

**Returns**: float - Combined coefficient (W/m²·K)

**Formula**: h = (h_forced³ + h_natural³)^(1/3)

##### radiation_heat_transfer_coefficient(T_surf: float, T_surr: float, emissivity: float) -> float
**Purpose**: Calculate linearized radiation heat transfer coefficient

**Parameters**:
- **T_surf**: float - Surface temperature (K)
- **T_surr**: float - Surrounding temperature (K)
- **emissivity**: float - Surface emissivity (0-1)

**Returns**: float - Radiation coefficient (W/m²·K)

**Formula**: h_rad = ε*σ*(T_surf⁴ - T_surr⁴)/(T_surf - T_surr)

##### condensation_heat_transfer_coefficient(wall_temp: float, vapor_temp: float, diameter: float, orientation: str = 'vertical') -> float
**Purpose**: Calculate condensation heat transfer coefficient

**Parameters**:
- **wall_temp**: float - Wall temperature (K)
- **vapor_temp**: float - Vapor temperature (K)
- **diameter**: float - Characteristic length (m)
- **orientation**: str = 'vertical' - Surface orientation

**Returns**: float - Condensation coefficient (W/m²·K)

**Notes**: Uses Nusselt's laminar film theory

##### pressure_drop(velocity: float, diameter: float, length: float, props: FlowProperties, roughness: float = 0.0) -> float
**Purpose**: Calculate pressure drop in pipe segment

**Parameters**:
- **velocity**: float - Flow velocity (m/s)
- **diameter**: float - Pipe diameter (m)
- **length**: float - Segment length (m)
- **props**: FlowProperties - Fluid properties
- **roughness**: float = 0.0 - Surface roughness (m)

**Returns**: float - Pressure drop (Pa)

##### solve_segment_heat_transfer(gas_temp: float, wall_temp: float, ambient_temp: float, velocity: float, segment_props: Dict, water_vapor_fraction: float = 0.01) -> Dict[str, Any]
**Purpose**: Main heat transfer solver for a segment

**Parameters**:
- **gas_temp**: float - Gas temperature (K)
- **wall_temp**: float - Wall temperature (K)
- **ambient_temp**: float - Ambient temperature (K)
- **velocity**: float - Gas velocity (m/s)
- **segment_props**: Dict - Segment geometry and properties
- **water_vapor_fraction**: float = 0.01 - Water vapor mole fraction

**Returns**: Dictionary containing:
- h_conv_inner: Inner convection coefficient
- h_conv_outer: Outer convection coefficient
- q_conv_inner: Inner convective heat flux
- q_conv_outer: Outer convective heat flux
- q_rad_outer: Outer radiative heat flux
- T_gas_new: Updated gas temperature
- T_wall_new: Updated wall temperature (single wall)
- T_liner_new, T_middle_new, T_outer_new: Triple wall temperatures
- reynolds_number: Flow Reynolds number
- nusselt_forced: Forced convection Nusselt
- nusselt_natural: Natural convection Nusselt
- pressure_drop: Segment pressure drop
- condensation_rate: Condensation rate if occurring

**Process**:
1. Calculate gas properties
2. Determine flow regime (Re, Gr, Ri)
3. Calculate heat transfer coefficients
4. Solve wall temperatures (single or triple)
5. Update gas temperature
6. Check for condensation

##### mass_transfer_coefficient(props: FlowProperties, diameter: float, reynolds: float, prandtl: float) -> float
**Purpose**: Calculate mass transfer coefficient using heat-mass analogy

**Parameters**:
- **props**: FlowProperties - Fluid properties
- **diameter**: float - Characteristic length (m)
- **reynolds**: float - Reynolds number
- **prandtl**: float - Prandtl number

**Returns**: float - Mass transfer coefficient (m/s)

**Method**: Chilton-Colburn analogy with Lewis number

##### buoyancy_pressure(height: float, rho_ambient: float, rho_gas: float) -> float
**Purpose**: Calculate buoyancy pressure difference

**Parameters**:
- **height**: float - Vertical height (m)
- **rho_ambient**: float - Ambient density (kg/m³)
- **rho_gas**: float - Gas density (kg/m³)

**Returns**: float - Buoyancy pressure (Pa)

**Formula**: ΔP = g*h*(ρ_ambient - ρ_gas)

##### air_gap_heat_transfer_coefficient(T_hot: float, T_cold: float, gap_thickness: float, orientation: str = 'vertical') -> float
**Purpose**: Calculate heat transfer coefficient for air gap between layers

**Parameters**:
- **T_hot**: float - Hot surface temperature (K)
- **T_cold**: float - Cold surface temperature (K)
- **gap_thickness**: float - Gap thickness (m)
- **orientation**: str = 'vertical' - Gap orientation

**Returns**: float - Heat transfer coefficient (W/m²·K)

**Notes**: Includes both conduction and natural convection in enclosed space

##### transient_wall_temperature(T_wall_old: float, T_gas: float, T_ambient: float, h_inner: float, h_outer: float, wall_props: Dict, dt: float, dx: float) -> float
**Purpose**: Calculate wall temperature for next time step

**Parameters**:
- **T_wall_old**: float - Current wall temperature (K)
- **T_gas**: float - Gas temperature (K)
- **T_ambient**: float - Ambient temperature (K)
- **h_inner**: float - Inner convection coefficient (W/m²·K)
- **h_outer**: float - Outer convection coefficient (W/m²·K)
- **wall_props**: Dict - Wall properties (k, rho, cp, thickness)
- **dt**: float - Time step (s)
- **dx**: float - Segment length (m)

**Returns**: float - New wall temperature (K)

**Method**: Explicit finite difference with energy balance

##### transient_triple_wall_temperatures(T_liner_old: float, T_middle_old: float, T_outer_old: float, T_gas: float, T_ambient: float, h_inner: float, h_outer: float, geometry: Dict, properties: Dict, dt: float) -> Tuple[float, float, float]
**Purpose**: Calculate triple wall temperatures using thermal network

**Parameters**:
- **T_liner_old**: float - Current liner temperature (K)
- **T_middle_old**: float - Current middle temperature (K)
- **T_outer_old**: float - Current outer temperature (K)
- **T_gas**: float - Gas temperature (K)
- **T_ambient**: float - Ambient temperature (K)
- **h_inner**: float - Inner convection coefficient (W/m²·K)
- **h_outer**: float - Outer convection coefficient (W/m²·K)
- **geometry**: Dict - Geometric parameters
- **properties**: Dict - Material properties for each layer
- **dt**: float - Time step (s)

**Returns**: Tuple of (T_liner_new, T_middle_new, T_outer_new)

**Method**: Thermal resistance network with capacitances

##### calculate_segment_heat_transfer(segment_data: Dict, dt: float) -> Dict
**Purpose**: Main interface for segment heat transfer calculations

**Parameters**:
- **segment_data**: Dict - All segment parameters and state
- **dt**: float - Time step (s)

**Returns**: Dictionary with updated temperatures and heat transfer rates

##### dew_point_temperature(pressure_Pa: float, humidity_ratio: float) -> float
**Purpose**: Calculate dew point temperature

**Parameters**:
- **pressure_Pa**: float - Total pressure (Pa)
- **humidity_ratio**: float - Humidity ratio (kg water/kg dry air)

**Returns**: float - Dew point temperature (K)

**Method**: Binary search for temperature where saturation pressure equals vapor pressure

---

## MODULE: coupled_solver.py - Flow-Thermal Coupling
Handles the iterative coupling between flow and heat transfer

### CLASS: ConvergenceMetrics (dataclass)
**Purpose**: Track convergence behavior during iterations

#### Attributes:
- **iteration**: int - Iteration count
- **temperature_residual**: float - Temperature change norm
- **velocity_residual**: float - Velocity change norm  
- **mass_flow_residual**: float - Mass flow change norm

### CLASS: CoupledFlowThermalSolver
**Purpose**: Iterative solver for coupled flow-thermal equations

#### Constructor:
```python
__init__(max_iterations: int = 20, tolerance: float = 1e-4, 
         relaxation_factor: float = 0.7)
```

**Parameters**:
- **max_iterations**: int = 20 - Maximum coupling iterations
- **tolerance**: float = 1e-4 - Convergence tolerance
- **relaxation_factor**: float = 0.7 - Under-relaxation factor

#### Methods:

##### solve_coupled_timestep(segment_index: int, inlet_conditions: Dict, current_state: Dict, boundary_conditions: Dict, geometry: Dict, dt: float, modules: Dict) -> Tuple[Dict, ConvergenceMetrics]
**Purpose**: Solve coupled flow-thermal problem for one segment

**Parameters**:
- **segment_index**: int - Segment number
- **inlet_conditions**: Dict - Contains:
  - T_gas_in: Inlet gas temperature
  - velocity_in: Inlet velocity
  - mass_flow_in: Inlet mass flow
- **current_state**: Dict - Current segment state
- **boundary_conditions**: Dict - Ambient conditions
- **geometry**: Dict - Segment geometry
- **dt**: float - Time step (s)
- **modules**: Dict - Physics modules

**Returns**: Tuple of (updated_state, convergence_metrics)

**Process**:
1. Initialize with current state
2. Iterate:
   - Calculate gas properties
   - Update buoyancy pressure
   - Solve for new velocity
   - Calculate heat transfer
   - Update temperatures
   - Check convergence
3. Apply relaxation if needed

##### calculate_residuals(old_state: Dict, new_state: Dict) -> Dict[str, float]
**Purpose**: Calculate convergence residuals

**Parameters**:
- **old_state**: Dict - Previous iteration state
- **new_state**: Dict - Current iteration state

**Returns**: Dictionary of residuals by variable

##### apply_relaxation(old_value: float, new_value: float, factor: Optional[float] = None) -> float
**Purpose**: Apply under-relaxation for stability

**Parameters**:
- **old_value**: float - Previous iteration value
- **new_value**: float - Current iteration value
- **factor**: Optional[float] - Relaxation factor (uses instance default if None)

**Returns**: float - Relaxed value

**Formula**: value = factor*new + (1-factor)*old

##### check_convergence(metrics: ConvergenceMetrics) -> bool
**Purpose**: Check if solution has converged

**Parameters**:
- **metrics**: ConvergenceMetrics - Current convergence state

**Returns**: bool - True if converged

**Criteria**: All residuals < tolerance

##### adapt_relaxation_factor(metrics_history: List[ConvergenceMetrics]) -> None
**Purpose**: Adjust relaxation factor based on convergence behavior

**Parameters**:
- **metrics_history**: List[ConvergenceMetrics] - Recent convergence history

**Side Effects**: Updates self.relaxation_factor

**Logic**:
- If oscillating: reduce factor
- If monotonic: increase factor
- Bounds: 0.3 <= factor <= 0.95

---

## MODULE: condensation_physics.py - Phase Change Modeling
Models water vapor condensation and latent heat effects

### CLASS: CondensationState (dataclass)
**Purpose**: Track condensation state at a point

#### Attributes:
- **is_condensing**: bool - Whether condensation is occurring
- **condensation_rate**: float - Mass condensation rate (kg/s)
- **film_thickness**: float - Liquid film thickness (m)
- **film_mass**: float - Accumulated condensate (kg)
- **latent_heat_rate**: float - Latent heat release (W)

### CLASS: CondensationPhysics
**Purpose**: Handle condensation calculations and thermodynamics

#### Constants:
- **M_WATER**: float = 0.018015 - Molecular weight of water (kg/mol)
- **M_AIR**: float = 0.028964 - Molecular weight of air (kg/mol)
- **R_WATER**: float = 461.5 - Gas constant for water vapor (J/kg·K)

#### Methods:

##### saturation_pressure(temperature: float) -> float
**Purpose**: Calculate water saturation pressure

**Parameters**:
- **temperature**: float - Temperature (K)

**Returns**: float - Saturation pressure (Pa)

**Method**: Antoine equation with coefficients for 273-373K range

##### latent_heat_vaporization(temperature: float) -> float
**Purpose**: Calculate latent heat of vaporization

**Parameters**:
- **temperature**: float - Temperature (K)

**Returns**: float - Latent heat (J/kg)

**Formula**: h_fg = 2501000 - 2370*(T - 273.15)

##### humidity_ratio_to_mole_fraction(humidity_ratio: float) -> float
**Purpose**: Convert humidity ratio to mole fraction

**Parameters**:
- **humidity_ratio**: float - kg water / kg dry air

**Returns**: float - Mole fraction of water vapor

**Formula**: x = ω/(ω + M_air/M_water)

##### mole_fraction_to_humidity_ratio(mole_fraction: float) -> float
**Purpose**: Convert mole fraction to humidity ratio

**Parameters**:
- **mole_fraction**: float - Mole fraction of water vapor

**Returns**: float - Humidity ratio (kg water/kg dry air)

**Formula**: ω = x*M_water/M_air/(1 - x)

##### calculate_condensation(gas_temp: float, wall_temp: float, pressure: float, water_vapor_fraction: float, heat_transfer_coeff: float, surface_area: float, dt: float, current_film_mass: float = 0.0) -> CondensationState
**Purpose**: Calculate condensation rate and effects

**Parameters**:
- **gas_temp**: float - Bulk gas temperature (K)
- **wall_temp**: float - Wall surface temperature (K)
- **pressure**: float - Total pressure (Pa)
- **water_vapor_fraction**: float - Mole fraction of water vapor
- **heat_transfer_coeff**: float - Convection coefficient (W/m²·K)
- **surface_area**: float - Condensing surface area (m²)
- **dt**: float - Time step (s)
- **current_film_mass**: float = 0.0 - Existing condensate (kg)

**Returns**: CondensationState with all results

**Process**:
1. Calculate dew point from vapor pressure
2. Check if wall_temp < dew_point
3. If condensing:
   - Calculate mass transfer coefficient
   - Determine condensation rate
   - Update film mass and thickness
   - Calculate latent heat release

##### mass_transfer_coefficient(heat_transfer_coeff: float, gas_temp: float, pressure: float) -> float
**Purpose**: Calculate mass transfer coefficient using heat-mass analogy

**Parameters**:
- **heat_transfer_coeff**: float - Heat transfer coefficient (W/m²·K)
- **gas_temp**: float - Gas temperature (K)
- **pressure**: float - Total pressure (Pa)

**Returns**: float - Mass transfer coefficient (m/s)

**Method**: Chilton-Colburn analogy with Lewis number ≈ 1

---

## MODULE: stable_triple_wall.py - Implicit Wall Solver
Numerically stable solver for triple-wall heat transfer with radiation

### CLASS: WallLayerProperties (dataclass)
**Purpose**: Material properties for one wall layer

#### Attributes:
- **thickness**: float - Layer thickness (m)
- **thermal_conductivity**: float - Thermal conductivity (W/m·K)
- **density**: float - Density (kg/m³)
- **specific_heat**: float - Specific heat (J/kg·K)
- **emissivity**: float = 0.0 - Surface emissivity (0-1)

### CLASS: StableTripleWallSolver
**Purpose**: Implicit solver for triple-wall systems with matrix scaling

#### Constants:
- **STEFAN_BOLTZMANN**: float = 5.67e-8 - Stefan-Boltzmann constant (W/m²·K⁴)

#### Methods:

##### calculate_view_factors(r_inner: float, r_outer: float) -> Tuple[float, float]
**Purpose**: Calculate radiation view factors for concentric cylinders

**Parameters**:
- **r_inner**: float - Inner radius (m)
- **r_outer**: float - Outer radius (m)

**Returns**: Tuple of (F_inner_to_outer, F_outer_to_inner)

**Notes**:
- F_inner_to_outer = 1.0 (all radiation hits outer)
- F_outer_to_inner = r_inner/r_outer (reciprocity)

##### radiation_resistance(r_inner: float, r_outer: float, emiss_inner: float, emiss_outer: float, length: float) -> float
**Purpose**: Calculate radiation resistance between surfaces

**Parameters**:
- **r_inner**: float - Inner radius (m)
- **r_outer**: float - Outer radius (m)
- **emiss_inner**: float - Inner surface emissivity
- **emiss_outer**: float - Outer surface emissivity
- **length**: float - Cylinder length (m)

**Returns**: float - Radiation resistance (K/W)

**Formula**: 
```
R_rad = (1-ε₁)/(ε₁A₁) + 1/A₁ + (1-ε₂)/(ε₂A₂)
```

##### scale_system(K_matrix: np.ndarray, C_matrix: np.ndarray, Q_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
**Purpose**: Scale matrices for better conditioning

**Parameters**:
- **K_matrix**: np.ndarray - Conductance matrix (W/K)
- **C_matrix**: np.ndarray - Capacitance matrix (J/K)
- **Q_vector**: np.ndarray - Heat source vector (W)

**Returns**: Tuple of (scaled_K, scaled_C, scaled_Q, scaling_factors)

**Method**: Diagonal scaling with S_ii = 1/√(K_ii*C_ii)

##### solve_triple_wall_stable(T_gas: float, T_ambient: float, h_inner: float, h_outer: float, T_liner_old: float, T_middle_old: float, T_outer_old: float, geometry: Dict, properties: Dict, dt: float) -> Tuple[float, float, float]
**Purpose**: Solve triple-wall heat transfer with implicit method

**Parameters**:
- **T_gas**: float - Gas temperature (K)
- **T_ambient**: float - Ambient temperature (K)
- **h_inner**: float - Inner convection coefficient (W/m²·K)
- **h_outer**: float - Outer convection coefficient (W/m²·K)
- **T_liner_old**: float - Previous liner temperature (K)
- **T_middle_old**: float - Previous middle temperature (K)
- **T_outer_old**: float - Previous outer temperature (K)
- **geometry**: Dict - Contains:
  - inner_diameter: Inner diameter (m)
  - length: Segment length (m)
- **properties**: Dict - Contains:
  - liner: WallLayerProperties
  - middle: WallLayerProperties
  - outer: WallLayerProperties
- **dt**: float - Time step (s)

**Returns**: Tuple of (T_liner_new, T_middle_new, T_outer_new)

**Process**:
1. Calculate geometry (radii, areas, volumes)
2. Compute thermal resistances:
   - Conduction through each layer
   - Radiation between layers
   - Convection at boundaries
3. Build conductance matrix K
4. Build capacitance matrix C
5. Scale system for conditioning
6. Solve: (C/dt + K)*T_new = C/dt*T_old + Q
7. Apply physical bounds

**Matrix Structure** (3x3 for three temperature nodes):
- Node 1: Inner liner surface
- Node 2: Middle layer (insulation)
- Node 3: Outer surface

##### linearize_radiation(T_ref: float, emissivity: float) -> float
**Purpose**: Linearize radiation for implicit solver

**Parameters**:
- **T_ref**: float - Reference temperature (K)
- **emissivity**: float - Surface emissivity

**Returns**: float - Linearized radiation coefficient (W/m²·K)

**Formula**: h_rad = 4*ε*σ*T³

---

## MODULE: adaptive_triple_wall.py - Adaptive Wall Solver
Adaptive time-stepping solver for stiff triple-wall problems

### CLASS: AdaptiveTripleWallSolver
**Purpose**: Handle triple-wall systems with automatic sub-stepping

#### Constants:
- **STEFAN_BOLTZMANN**: float = 5.67e-8 - Stefan-Boltzmann constant

#### Methods:

##### solve_with_adaptive_dt(T_gas: float, T_ambient: float, h_inner: float, h_outer: float, T_liner_old: float, T_middle_old: float, T_outer_old: float, geometry: Dict, properties: Dict, dt_global: float) -> Tuple[float, float, float]
**Purpose**: Solve with automatic sub-stepping for stability

**Parameters**: Same as stable solver except:
- **dt_global**: float - Global time step (can be large)

**Returns**: Tuple of (T_liner_new, T_middle_new, T_outer_new)

**Process**:
1. Calculate thermal time constants:
   - τ = ρ*cp*δ²/k for each layer
2. Determine required timestep:
   - dt_required = 0.1*min(τ₁, τ₂, τ₃)
3. If dt_global <= dt_required:
   - Use single step
4. Else:
   - n_substeps = ceil(dt_global/dt_required)
   - dt_sub = dt_global/n_substeps
   - Loop through substeps

##### _solve_single_step(T_gas: float, T_ambient: float, h_inner: float, h_outer: float, T_liner_old: float, T_middle_old: float, T_outer_old: float, geometry: Dict, properties: Dict, dt: float) -> Tuple[float, float, float]
**Purpose**: Solve single timestep using semi-implicit method

**Parameters**: Same as main solver

**Returns**: Tuple of updated temperatures

**Method**: Semi-implicit update
- Liner uses explicit gas temperature
- Middle uses new liner temperature
- Outer uses new middle temperature

**Notes**: 
- More stable than fully explicit
- Less complex than fully implicit
- No radiation between layers (simplified)

---

## MODULE: energy_diagnostics.py - Conservation Checks
Monitors energy conservation throughout simulation

### CLASS: EnergyDiagnostics
**Purpose**: Track and verify energy conservation

#### Methods:

##### check_energy_balance(time_step: int, dt: float, gas_temps: np.ndarray, wall_temps: np.ndarray, mass_flow_rate: float, heat_fluxes: np.ndarray, segment_volumes: np.ndarray, segment_areas: np.ndarray, inlet_temp: float, ambient_temp: float, densities: np.ndarray) -> Dict[str, float]
**Purpose**: Perform energy balance check for current timestep

**Parameters**:
- **time_step**: int - Current time index
- **dt**: float - Time step (s)
- **gas_temps**: np.ndarray - Gas temperatures array
- **wall_temps**: np.ndarray - Wall temperatures array
- **mass_flow_rate**: float - Flow rate (kg/s)
- **heat_fluxes**: np.ndarray - Wall heat flux array (W/m²)
- **segment_volumes**: np.ndarray - Gas volumes (m³)
- **segment_areas**: np.ndarray - Surface areas (m²)
- **inlet_temp**: float - Inlet temperature (K)
- **ambient_temp**: float - Ambient temperature (K)
- **densities**: np.ndarray - Gas densities (kg/m³)

**Returns**: Dictionary containing:
- energy_in_rate: Energy input rate (W)
- energy_stored_rate: Storage rate (W)
- energy_lost_rate: Loss rate (W)
- balance: Imbalance (W)
- relative_error: balance/energy_in

**Energy Components**:
1. Input: ṁ*cp*(T_in - T_ref)
2. Storage: Σ(ρ*V*cp*dT/dt) for gas and walls
3. Losses: Σ(q*A) to ambient

##### calculate_cumulative_energy(all_diagnostics: List[Dict], dt: float) -> Dict[str, float]
**Purpose**: Integrate energy over time

**Parameters**:
- **all_diagnostics**: List[Dict] - Energy data for all timesteps
- **dt**: float - Time step (s)

**Returns**: Dictionary with cumulative energies (J)

##### generate_report(all_diagnostics: List[Dict]) -> str
**Purpose**: Generate formatted energy balance report

**Parameters**:
- **all_diagnostics**: List[Dict] - All diagnostic data

**Returns**: str - Formatted report with statistics

**Report Includes**:
- Average energy rates
- Maximum imbalance
- Time of maximum error
- Cumulative totals

---

## MODULE: early_diagnostics.py - Startup Diagnostics
Monitors simulation startup for numerical issues

### CLASS: EarlyTimeDiagnostics
**Purpose**: Detect problems in first few timesteps

#### Constructor:
```python
__init__(verbose: bool = False)
```

**Parameters**:
- **verbose**: bool = False - Enable detailed output

#### Methods:

##### check_timestep(step: int, current_state: Dict, previous_state: Dict, dt: float) -> Dict[str, Any]
**Purpose**: Check for numerical problems at each timestep

**Parameters**:
- **step**: int - Time step number
- **current_state**: Dict - Current system state
- **previous_state**: Dict - Previous system state
- **dt**: float - Time step (s)

**Returns**: Dictionary containing:
- timestep: Current step number
- warnings: List of warning messages
- errors: List of error messages
- metrics: Numerical metrics

**Checks Performed**:
1. Temperature bounds
2. Temperature change rates
3. Mass flow stability
4. Pressure consistency
5. NaN/Inf detection

##### check_temperature_bounds(temps: np.ndarray, ambient_temp: float, max_temp: float = 2000) -> List[str]
**Purpose**: Check for unphysical temperatures

**Parameters**:
- **temps**: np.ndarray - Temperature array
- **ambient_temp**: float - Ambient temperature (K)
- **max_temp**: float = 2000 - Maximum allowed (K)

**Returns**: List[str] - Warning messages

**Checks**:
- T < ambient (unless cooling)
- T > max_temp (safety limit)
- T < 0 (absolute zero)

##### check_convergence_quality(convergence_metrics: List[Any]) -> Dict[str, Any]
**Purpose**: Analyze coupling convergence behavior

**Parameters**:
- **convergence_metrics**: List[Any] - Convergence history

**Returns**: Dictionary with convergence statistics

##### diagnose_instability(state_history: List[Dict]) -> str
**Purpose**: Identify cause of numerical instability

**Parameters**:
- **state_history**: List[Dict] - Recent state history

**Returns**: str - Diagnostic message with recommendations

**Analysis**:
- Oscillation detection
- Divergence patterns
- Suggested fixes (reduce dt, adjust relaxation)

---

## MODULE: draft_analysis.py - Draft Performance Analysis

### Purpose
Analyzes chimney draft (buoyancy-driven pressure) and visualizes performance metrics. Calculates theoretical draft, pressure losses, and actual flow performance.

### Classes

#### DraftResults (dataclass)
**Purpose**: Container for draft calculation results

**Attributes**:
- **time**: np.ndarray - Time array (s)
- **total_draft**: np.ndarray - Total theoretical draft pressure (Pa)
- **segment_draft**: np.ndarray - Draft contribution by segment (Pa)
- **available_draft**: np.ndarray - Draft after losses (Pa)
- **draft_velocity**: np.ndarray - Theoretical velocity from draft (m/s)
- **pressure_losses**: np.ndarray - Friction and fitting losses (Pa)
- **mass_flow_rate**: np.ndarray - Actual mass flow (kg/s)
- **stack_height**: float - Total chimney height (m)

#### DraftAnalyzer
**Purpose**: Calculate and visualize chimney draft performance

**Attributes**:
- **chimney**: Chimney - Reference to chimney object
- **g**: float = 9.81 - Gravitational acceleration (m/s²)

### Methods

##### __init__(self, chimney: Chimney)
**Purpose**: Initialize analyzer with chimney reference

**Parameters**:
- **chimney**: Chimney - Chimney object to analyze

##### calculate_draft(self) -> DraftResults
**Purpose**: Calculate draft pressure for all timesteps

**Returns**: DraftResults object with complete draft analysis

**Calculations**:
1. Segment-by-segment buoyancy: ΔP = g·h·(ρ_ambient - ρ_gas)
2. Pressure losses: friction + fittings
3. Available draft = total draft - losses
4. Theoretical velocity from Bernoulli

##### _friction_factor(self, Re: float, relative_roughness: float) -> float
**Purpose**: Calculate Darcy friction factor

**Parameters**:
- **Re**: float - Reynolds number
- **relative_roughness**: float - ε/D ratio

**Returns**: float - Friction factor

**Correlations**:
- Laminar (Re < 2300): f = 64/Re
- Smooth turbulent: Blasius equation
- Rough turbulent: Colebrook approximation

##### plot_draft_analysis(self, results: DraftResults, save_path: Optional[str] = None)
**Purpose**: Create comprehensive 2x2 draft analysis plots

**Parameters**:
- **results**: DraftResults - Calculation results
- **save_path**: Optional[str] - Path to save figure

**Returns**: matplotlib.figure.Figure

**Subplots**:
1. Draft vs time (total, available, losses)
2. Draft contribution by segment
3. Mass flow and velocity vs time
4. Draft efficiency with steady-state values

##### plot_draft_vs_temperature(self, results: DraftResults, save_path: Optional[str] = None)
**Purpose**: Plot draft-temperature relationship

**Parameters**:
- **results**: DraftResults - Calculation results
- **save_path**: Optional[str] - Path to save figure

**Returns**: matplotlib.figure.Figure

**Subplots**:
1. Total draft vs average gas temperature
2. Temperature profiles at steady state

##### print_draft_summary(self, results: DraftResults)
**Purpose**: Print comprehensive text summary

**Parameters**:
- **results**: DraftResults - Calculation results

**Output Sections**:
1. Chimney configuration
2. Steady-state results
3. Flow performance
4. Temperature conditions
5. Required draft for target flows

---

## MODULE: chimney_geometry_plot.py - Geometry Visualization

### Purpose
Creates visual representations of chimney geometry with temperature overlays. Supports both static snapshots and animated sequences showing temperature evolution.

### Functions

#### plot_chimney_geometry(chimney, timesteps_to_plot: Optional[List[int]] = None, save_animation: bool = False, fps: int = 10, figsize: Tuple[float, float] = (10, 8), wall_scale: float = 5.0)
**Purpose**: Create animated visualization of chimney geometry with temperature evolution

**Parameters**:
- **chimney**: Chimney - Chimney object with simulation results
- **timesteps_to_plot**: Optional[List[int]] - Timestep indices to animate (default: 20 frames)
- **save_animation**: bool = False - Save as animated GIF
- **fps**: int = 10 - Animation frames per second
- **figsize**: Tuple[float, float] = (10, 8) - Figure dimensions
- **wall_scale**: float = 5.0 - Wall thickness scaling factor for visibility

**Features**:
- Handles both vertical and horizontal sections
- Shows gas flow with arrows
- Temperature-based coloring
- Animated time progression

#### plot_chimney_snapshot(chimney, timestep: int, figsize: Tuple[float, float] = (10, 8), wall_scale: float = 5.0, save_path: Optional[str] = None)
**Purpose**: Plot single timestep of chimney geometry with temperatures

**Parameters**:
- **chimney**: Chimney - Chimney object with simulation results
- **timestep**: int - Specific timestep to visualize
- **figsize**: Tuple[float, float] = (10, 8) - Figure dimensions
- **wall_scale**: float = 5.0 - Wall thickness scaling factor
- **save_path**: Optional[str] - Path to save figure

**Features**:
- Static visualization at specific time
- Temperature colormap with scale
- Gas flow indicators
- Geometry cross-section view

---

## DATA STRUCTURES

### Input Configuration Format

#### Single Wall Example:
```python
{
    'type': 'single',
    'inner_diameter_ft': 6/12,      # 6 inches
    'height_ft': 20,
    'z_ft': 0,
    'thickness_w_m': 0.005,         # 5mm steel
    'orientation': 'vertical',
    'density_w_kg_m3': 7850,        # Steel
    'cp_w_J_kg_K': 460,
    'k_w_W_m_K': 50,
    'epsilon_w': 0.8,
    'T_ambient_K': 293.15,
    'Pa': 101325,
    'num_segments': 20,
    'water_vapor_fraction': 0.01,   # Optional
    'roughness': 0.0                # Optional
}
```

#### Triple Wall Example:
```python
{
    'type': 'triple',
    'inner_diameter_ft': 8/12,
    'height_ft': 15,
    'z_ft': 5,                      # 5 ft elevation
    'thickness_liner_m': 0.003,     # 3mm steel liner
    'thickness_insulation_m': 0.025, # 25mm insulation
    'thickness_outer_m': 0.001,     # 1mm aluminum
    'orientation': 'vertical',
    'density_liner_kg_m3': 7850,
    'density_insulation_kg_m3': 128,
    'density_outer_kg_m3': 2700,
    'cp_liner_J_kg_K': 460,
    'cp_insulation_J_kg_K': 840,
    'cp_outer_J_kg_K': 900,
    'k_liner_W_m_K': 50,
    'k_insulation_W_m_K': 0.04,
    'k_outer_W_m_K': 200,
    'epsilon_liner': 0.8,
    'epsilon_insulation': 0.9,
    'epsilon_outer': 0.2,
    'T_ambient_K': 293.15,
    'Pa': 101325,
    'num_segments': 15
}
```

### Output Data Arrays

All arrays are numpy arrays with shape [time_steps+1, total_segments]:

```python
# Temperature arrays (K)
T_gas[t, s]              # Gas temperature
T_wall[t, s]             # Wall inner surface
T_outer[t, s]            # Wall outer surface
T_liner[t, s]            # Triple wall liner
T_insulation[t, s]       # Triple wall middle
dew_point_temperature[t, s]  # Condensation threshold

# Flow arrays
velocity[t, s]           # Gas velocity (m/s)
mass_flow_rate[t]        # System flow rate (kg/s)
density[t, s]            # Gas density (kg/m³)

# Heat transfer arrays
heat_flux_wall[t, s]     # Wall heat flux (W/m²)
h_conv_inner[t, s]       # Inner convection (W/m²·K)
h_conv_outer[t, s]       # Outer convection (W/m²·K)

# Pressure arrays
pressure_drop[t, s]      # Segment ΔP (Pa)
draft_pressure[t, s]     # Cumulative draft (Pa)

# Condensation arrays
water_vapor_mass_fraction[t, s]  # Humidity
condensation_rate[t, s]          # Rate (kg/s)
```

### Summary Output Format

```python
{
    'total_segments': 50,
    'total_time': 600.0,                    # seconds
    'time_steps': 600,
    'final_gas_temp_range_C': (45.2, 250.3),  # (min, max) in °C
    'final_wall_temp_range_C': (25.1, 180.5), # (min, max) in °C
    'final_mass_flow_rate_kg_s': 0.0234,       # kg/s
    'parts_config': [
        {
            'type': 'single',
            'height_m': 6.096,              # 20 ft
            'diameter_m': 0.1524,           # 6 inches
            'segments': 20
        },
        {
            'type': 'triple', 
            'height_m': 4.572,              # 15 ft
            'diameter_m': 0.2032,           # 8 inches
            'segments': 30
        }
    ]
}
```

---

## USAGE EXAMPLES

### Basic Simulation
```python
from flued_module import Chimney

# Single wall chimney
parts = [{
    'type': 'single',
    'inner_diameter_ft': 6/12,
    'height_ft': 20,
    'z_ft': 0,
    'thickness_w_m': 0.005,
    'orientation': 'vertical',
    'density_w_kg_m3': 7850,
    'cp_w_J_kg_K': 460,
    'k_w_W_m_K': 50,
    'epsilon_w': 0.8,
    'T_ambient_K': 293.15,
    'Pa': 101325,
    'num_segments': 20
}]

# Run simulation
chimney = Chimney(parts=parts, dt=1.0, total_time=600)
chimney.run_simulation(save_interval=10)

# Get results
summary = chimney.get_summary_data()
print(f"Outlet temp: {summary['outlet_temperature']:.1f} K")
```

### Multi-Section Chimney
```python
# Bottom single wall, top insulated
bottom = {
    'type': 'single',
    'inner_diameter_ft': 8/12,
    'height_ft': 5,
    'z_ft': 0,
    # ... properties
}

top = {
    'type': 'triple',
    'inner_diameter_ft': 8/12,
    'height_ft': 15,
    'z_ft': 5,
    # ... properties
}

chimney = Chimney(parts=[bottom, top], dt=0.5, total_time=1200)
chimney.run_simulation()
```

### Custom Analysis
```python
# Plot results
chimney.plot_temperature_profile([0, 100, 300, 600])

# Check energy balance
energy = chimney.calculate_energy_balance()
print(f"Energy error: {energy['relative_error']:.1%}")

# Find condensation
for seg in range(chimney.total_segments):
    if chimney.condensation_rate[-1, seg] > 0:
        h = chimney.segment_heights[seg]
        print(f"Condensation at {h:.1f} m")

# Export data
import pandas as pd
df = pd.DataFrame({
    'Height_m': chimney.segment_heights,
    'T_gas': chimney.T_gas[-1, :],
    'T_wall': chimney.T_wall[-1, :]
})
df.to_csv('results.csv')
```

---

## ERROR HANDLING

### Common Errors

1. **ValueError: Time step too large**
   - Reduce dt for stability
   - Use adaptive solver for triple walls

2. **ValueError: Invalid geometry**
   - Check dimensions are positive
   - Verify units (feet vs meters)

3. **RuntimeWarning: Invalid value in log**
   - Check for negative temperatures
   - Verify pressure is positive

4. **LinAlgError: Matrix singular**
   - Triple wall properties may be extreme
   - Check thermal conductivities

### Error Recovery
```python
try:
    chimney.run_simulation()
except ValueError as e:
    if "Time step" in str(e):
        # Retry with smaller timestep
        chimney = Chimney(parts, dt=0.1, total_time)
        chimney.run_simulation()
```

---

## PERFORMANCE OPTIMIZATION

### Memory Usage
- Arrays scale as O(time_steps × segments)
- 10 min simulation, 50 segments ≈ 50 MB

### Computation Time
- Scales as O(time_steps × segments × iterations)
- Coupling iterations: 3-5 typical
- Triple wall: 2-3x slower than single

### Optimization Tips
1. Start with coarse resolution
2. Use larger dt with adaptive solver
3. Disable coupling for preliminary runs
4. Reduce segments in uniform sections

---

## EXTENSION GUIDE

### Adding New Correlations
1. Subclass ComprehensiveHeatTransfer
2. Override specific methods
3. Use in Chimney constructor

### Adding Wall Types
1. Create new solver module
2. Follow stable/adaptive pattern
3. Add to chimney._time_step logic

### Custom Diagnostics
1. Add methods to energy_diagnostics
2. Call from chimney._time_step
3. Store in results arrays

---

## DEPENDENCIES
- numpy >= 1.20.0
- scipy >= 1.7.0 (optional, for advanced features)
- matplotlib >= 3.4.0 (for plotting)
- Python >= 3.7 (for dataclasses)