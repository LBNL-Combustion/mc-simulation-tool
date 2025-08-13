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
- **convert_from_imperial(parts_imperial: List[Dict]) -> List[ChimneyPart]**: Static method to convert imperial units to SI

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

##### _default_inlet_temp_fire(self, t_sec: float) -> float
**Purpose**: Calculate inlet temperature using ISO 834 fire curve

**Parameters**:
- **t_sec**: float - Time in seconds since ignition

**Returns**: float - Temperature in Kelvin

**Notes**: 
- Uses ISO 834: T = 20 + 345*log10(8t + 1) where t is in minutes
- Scales to 50% for flue gases (empirical)
- Caps at 800K for safety

##### _solve_comprehensive_heat_transfer(self, step: int, T_inlet: float) -> None
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

##### _update_flow_and_pressure(self, step: int) -> None
**Purpose**: Update mass flow rate and pressure distribution

**Parameters**:
- **step**: int - Current time step index

**Process**:
1. Calculate total buoyancy pressure
2. Calculate total flow resistance
3. Solve for mass flow rate
4. Update velocities in each segment
5. Calculate pressure drops

##### get_summary_data(self) -> Dict[str, Any]
**Purpose**: Generate summary statistics and final results

**Returns**: Dictionary containing:
- total_segments: Total number of segments
- total_time: Simulation duration (s)
- time_steps: Number of time steps
- final_gas_temp_range_C: Tuple of (min, max) gas temps in °C
- final_wall_temp_range_C: Tuple of (min, max) wall temps in °C
- final_mass_flow_rate_kg_s: Final mass flow rate (kg/s)
- parts_config: List of part configurations with type, height, diameter, segments

##### plot_temperature_profiles(self, times_to_plot: Optional[List[float]] = None) -> None
**Purpose**: Plot temperature vs height at specified times

**Parameters**:
- **times_to_plot**: Optional[List[float]] - Times in seconds to plot
  - If None, plots at t=0, 25%, 50%, 75%, 100% of total time

**Side Effects**: 
- Creates matplotlib figure showing gas and wall temperatures vs height

##### _check_energy_conservation(self) -> Dict[str, float]
**Purpose**: Check energy conservation at final time

**Returns**: Dictionary containing:
- energy_in_total: Total energy input (J)
- energy_stored_gas: Energy stored in gas (J)
- energy_stored_walls: Energy stored in walls (J)
- energy_lost_total: Energy lost to ambient (J)
- energy_balance: Input - stored - lost (J)
- percent_error: 100 * balance / input

**Note**: Called internally by check_conservation()

##### _initialize_arrays(self) -> None
**Purpose**: Pre-allocate all numpy arrays for simulation
**Side Effects**: Creates all state and geometry arrays

##### _set_initial_conditions(self) -> None
**Purpose**: Set initial temperatures and flow conditions
**Side Effects**: Fills arrays at time index 0

##### check_conservation(self) -> None
**Purpose**: Check mass and energy conservation
**Side Effects**: Prints detailed conservation diagnostics

##### analyze_draft(self, plot: bool = True, save_plots: bool = False, plot_dir: str = './') -> Dict
**Purpose**: Analyze chimney draft performance
**Parameters**:
- **plot**: bool = True - Whether to create plots
- **save_plots**: bool = False - Whether to save plots to disk
- **plot_dir**: str = './' - Directory for saving plots
**Returns**: Dictionary with draft analysis results

##### plot_chimney_geometry(self, timesteps_to_plot=None, save_animation=False, animation_filename='chimney_animation.gif') -> None
**Purpose**: Visualize chimney geometry and temperature distribution
**Parameters**:
- **timesteps_to_plot**: List of timestep indices to plot
- **save_animation**: bool = False - Whether to save as GIF
- **animation_filename**: str - Name of animation file

##### plot_segment_diagnostics(self) -> None
**Purpose**: Create detailed diagnostic plots for each segment
**Side Effects**: Creates multi-panel figure with temperatures, velocities, heat transfer coefficients

##### _default_inlet_temp(self, t: float, T_ambient: float = 293.15, T_max: float = 773.15, tau: float = 300) -> float
**Purpose**: Calculate inlet temperature with exponential ramp-up
**Parameters**:
- **t**: float - Time in seconds
- **T_ambient**: float = 293.15 - Starting ambient temperature (K)
- **T_max**: float = 773.15 - Maximum temperature (K) 
- **tau**: float = 300 - Time constant (s), temperature reaches 63% of max at t=tau
**Returns**: float - Inlet temperature (K)
**Formula**: T = T_ambient + (T_max - T_ambient) * (1 - exp(-t/tau))

##### get_segment_info(segment_idx: int) -> tuple
**Purpose**: Get part index and local segment index for a global segment
**Parameters**:
- **segment_idx**: int - Global segment index (0 to total_segments-1)
**Returns**: Tuple of (part_idx, local_segment_idx)

##### temp_evolution_by_part(self, ambient_temp: bool = False) -> None
**Purpose**: Plot temperature evolution over time for each chimney part
**Parameters**:
- **ambient_temp**: bool = False - Whether to show ambient temperature line
**Side Effects**: Creates figure with subplots for each part

##### temp_evolution_br(self) -> None
**Purpose**: Plot temperature evolution at bottom and top of chimney
**Side Effects**: Creates figure showing inlet/outlet temperatures over time

##### plot_draft(self) -> None
**Purpose**: Plot draft pressure distribution along chimney height
**Side Effects**: Creates figure showing pressure vs height at different times

##### plot_buoyancy_pressure(self) -> None
**Purpose**: Plot buoyancy pressure distribution
**Side Effects**: Creates figure showing buoyancy driving pressure vs height

##### temp_evolution_by_segment(self) -> None
**Purpose**: Plot temperature evolution for each segment (creates many subplots)
**Side Effects**: Creates large figure with subplot for each segment

##### _plot_heat_transfer_coefficients(self) -> None
**Purpose**: Plot heat transfer coefficients along chimney
**Side Effects**: Creates figure showing h_inner and h_outer vs height

##### _print_detailed_segment_info(self) -> None
**Purpose**: Print detailed information about each segment
**Side Effects**: Prints table with geometry and properties

##### _check_mass_conservation(self) -> Dict[str, float]
**Purpose**: Check mass conservation through the chimney
**Returns**: Dictionary with mass flow statistics
**Note**: Called internally by check_conservation()

##### plot_chimney_snapshot(self, timestep: int = -1, figsize: tuple = (10, 8), wall_scale: float = 5.0, show_temps: bool = True, show_grid: bool = True, save_fig: bool = False, filename: str = 'chimney_snapshot.png') -> None
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
Physical properties and constants for flue gas and air calculations

### CLASS: PhysicalProperties
**Purpose**: Physical properties and correlations for flue gas and air

#### Constants:
- **R_UNIVERSAL**: float = 8314.0 - Universal gas constant J/(kmol·K)
- **R_AIR**: float = 287.0 - Specific gas constant for air J/(kg·K)
- **GRAVITY**: float = 9.81 - Gravitational acceleration m/s²
- **T_STD**: float = 273.15 - Standard temperature in K (0°C)
- **P_STD**: float = 101325.0 - Standard pressure in Pa (1 atm)

#### Methods:

##### air_density(temperature_K, pressure_Pa=P_STD)
**Purpose**: Calculate air density using ideal gas law

**Parameters**:
- **temperature_K**: Temperature in Kelvin
- **pressure_Pa**: Pressure in Pascals (default is P_STD)

**Returns**: Density in kg/m³

**Formula**: ρ = P/(R_AIR * T)

##### air_viscosity(temperature_K)
**Purpose**: Calculate air dynamic viscosity using Sutherland's formula

**Parameters**:
- **temperature_K**: Temperature in Kelvin

**Returns**: Dynamic viscosity in Pa·s

**Formula**: μ = μ_ref * (T/T_ref)^1.5 * (T_ref + S)/(T + S)
- μ_ref = 1.716e-5 Pa·s at T_ref = 273.15 K
- S = 110.4 K (Sutherland constant)

##### air_thermal_conductivity(temperature_K)
**Purpose**: Calculate air thermal conductivity

**Parameters**:
- **temperature_K**: Temperature in Kelvin

**Returns**: Thermal conductivity in W/(m·K)

**Formula**: k = 2.624e-3 * T^1.5 / (T + 245.4 * 10^(-12/T))

##### air_specific_heat(temperature_K)
**Purpose**: Calculate air specific heat at constant pressure

**Parameters**:
- **temperature_K**: Temperature in Kelvin

**Returns**: Specific heat in J/(kg·K)

**Formula**: cp = 1030.5 - 0.19975*T + 4.2775e-4*T² - 2.4223e-7*T³

##### nusselt_natural_convection_vertical_cylinder(rayleigh, height_diameter_ratio)
**Purpose**: Calculate Nusselt number for natural convection on a vertical cylinder

**Parameters**:
- **rayleigh**: Rayleigh number
- **height_diameter_ratio**: H/D ratio

**Returns**: Nusselt number (dimensionless)

**Correlations**:
- Ra < 10⁹: Nu = 0.59 * Ra^0.25
- Ra >= 10⁹: Nu = 0.1 * Ra^(1/3)

##### friction_factor_smooth_tube(reynolds)
**Purpose**: Calculate friction factor for smooth tubes

**Parameters**:
- **reynolds**: Reynolds number

**Returns**: Friction factor (dimensionless)

**Correlations**:
- Re < 2300 (laminar): f = 64/Re
- Re >= 2300 (turbulent): f = 0.316/Re^0.25

---

## MODULE: mass_flow.py - Flow and Pressure Calculations
Mass flow and pressure calculations for chimney simulation

### CLASS: MassFlow
**Purpose**: Mass flow and pressure calculations for flue systems

#### Constructor:
```python
__init__(self)
```
**Note**: Initializes with PhysicalProperties instance

#### Methods:

##### buoyancy_pressure(self, height, density_cold, density_hot)
**Purpose**: Calculate buoyancy pressure difference

**Parameters**:
- **height**: Height difference (m)
- **density_cold**: Density of cold fluid (kg/m³)
- **density_hot**: Density of hot fluid (kg/m³)

**Returns**: Buoyancy pressure (Pa)

**Formula**: ΔP = (ρ_cold - ρ_hot) * g * height

##### pressure_loss_friction(self, friction_factor, length, diameter, density, velocity)
**Purpose**: Calculate pressure loss due to friction

**Parameters**:
- **friction_factor**: Darcy friction factor (dimensionless)
- **length**: Pipe length (m)
- **diameter**: Pipe diameter (m)
- **density**: Fluid density (kg/m³)
- **velocity**: Flow velocity (m/s)

**Returns**: Pressure loss (Pa)

**Formula**: ΔP = f * (L/D) * (ρv²/2)

##### pressure_loss_minor(self, loss_coefficient, density, velocity)
**Purpose**: Calculate minor pressure losses

**Parameters**:
- **loss_coefficient**: Minor loss coefficient (dimensionless)
- **density**: Fluid density (kg/m³)
- **velocity**: Flow velocity (m/s)

**Returns**: Pressure loss (Pa)

**Formula**: ΔP = K * (ρv²/2)

##### entrance_loss_coefficient(self, geometry='sharp')
**Purpose**: Get entrance loss coefficient

**Parameters**:
- **geometry**: 'sharp', 'rounded', or 'bell_mouth'

**Returns**: Loss coefficient (dimensionless)

**Values**:
- sharp: 0.5
- rounded: 0.1
- bell_mouth: 0.05

##### exit_loss_coefficient(self)
**Purpose**: Get exit loss coefficient

**Returns**: Loss coefficient = 1.0

##### elbow_loss_coefficient(self, angle_degrees, diameter, radius_curvature=None)
**Purpose**: Calculate elbow loss coefficient

**Parameters**:
- **angle_degrees**: Bend angle in degrees
- **diameter**: Pipe diameter (m)
- **radius_curvature**: Radius of curvature (m), if None uses 1.5*diameter

**Returns**: Loss coefficient (dimensionless)

**Calculation**:
- Base K_90 depends on r/d ratio
- Scaled by (angle/90)^1.5

##### calculate_mass_flow_rate(self, driving_pressure, total_resistance, density)
**Purpose**: Calculate mass flow rate from driving pressure and system resistance

**Parameters**:
- **driving_pressure**: Net driving pressure (Pa)
- **total_resistance**: Total system flow resistance (Pa·s²/kg²)
- **density**: Average fluid density (kg/m³)

**Returns**: Mass flow rate (kg/s)

**Formula**: ṁ = √(ΔP / R_total)

##### system_resistance_analysis(self, segments_data)
**Purpose**: Calculate total system flow resistance

**Parameters**:
- **segments_data**: List of dictionaries with segment properties
  - Each dict should contain: length, diameter, friction_factor, minor_losses

**Returns**: Total flow resistance (Pa·s²/kg²)

**Calculation**:
- Friction resistance: f*L/(D*A²)
- Minor resistance: ΣK/A²

##### draft_calculation(self, chimney_height, T_gas_avg, T_ambient, pressure_ambient)
**Purpose**: Calculate natural draft pressure

**Parameters**:
- **chimney_height**: Total chimney height (m)
- **T_gas_avg**: Average gas temperature in chimney (K)
- **T_ambient**: Ambient temperature (K)
- **pressure_ambient**: Ambient pressure (Pa)

**Returns**: Draft pressure (Pa)

##### wind_pressure_effect(self, wind_speed, wind_direction_deg, chimney_height, building_height=None, pressure_coefficients=None)
**Purpose**: Calculate wind pressure effects on chimney draft

**Parameters**:
- **wind_speed**: Wind speed (m/s)
- **wind_direction_deg**: Wind direction relative to building (degrees)
- **chimney_height**: Chimney height above ground (m)
- **building_height**: Building height (m), default is chimney_height * 0.5
- **pressure_coefficients**: Dict with 'windward' and 'leeward' coefficients

**Returns**: Wind pressure difference (Pa)

**Default coefficients**:
- windward: 0.8
- leeward: -0.5

##### flow_network_solver(self, nodes, connections, boundary_conditions, max_iterations=50, tolerance=1e-6)
**Purpose**: Solve flow network using iterative method

**Parameters**:
- **nodes**: Dictionary of node properties
- **connections**: List of connection dictionaries with flow resistance
- **boundary_conditions**: Dictionary of known pressures or flows
- **max_iterations**: Maximum number of iterations
- **tolerance**: Convergence tolerance for pressure residuals

**Returns**: Dictionary containing:
- pressures: Node pressures array
- flows: Connection flows array
- iterations: Number of iterations
- converged: Boolean convergence status

##### equivalent_leakage_area(self, flow_rate, pressure_difference, reference_pressure=10.0, exponent=0.5)
**Purpose**: Calculate equivalent leakage area

**Parameters**:
- **flow_rate**: Flow rate (m³/s)
- **pressure_difference**: Pressure difference (Pa)
- **reference_pressure**: Reference pressure for ELA calculation (Pa)
- **exponent**: Flow exponent (typically 0.5 for turbulent flow)

**Returns**: Equivalent leakage area (m²)

---

## MODULE: comprehensive_heat_transfer.py - Heat Transfer Solver
Comprehensive heat transfer calculations including natural, forced, and mixed convection, radiation, and condensation

### CLASS: FlowProperties (dataclass)
**Purpose**: Properties for flow calculations

#### Attributes:
- **velocity**: float - Flow velocity (m/s)
- **density**: float - Density (kg/m³)
- **viscosity**: float - Dynamic viscosity (Pa·s)
- **thermal_conductivity**: float - Thermal conductivity (W/m·K)
- **specific_heat**: float - Specific heat (J/kg·K)
- **temperature**: float - Temperature (K)
- **pressure**: float - Pressure (Pa)

### CLASS: ComprehensiveHeatTransfer
**Purpose**: Comprehensive heat transfer model including natural/forced/mixed convection, radiation, condensation

#### Constructor:
```python
__init__(self)
```

**Note**: Initializes with instances of PhysicalProperties, CondensationPhysics, StableTripleWallSolver, AdaptiveTripleWallSolver, and EnergyDiagnostics

#### Methods:

##### gas_properties(self, temperature_K: float, pressure_Pa: float = 101325) -> FlowProperties
**Purpose**: Calculate gas properties at given temperature and pressure

**Parameters**:
- **temperature_K**: float - Temperature (K)
- **pressure_Pa**: float = 101325 - Pressure (Pa)

**Returns**: FlowProperties object with all fluid properties

##### reynolds_number(self, velocity: float, diameter: float, props: FlowProperties) -> float
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

##### prandtl_number(self, props: FlowProperties) -> float
**Purpose**: Calculate Prandtl number

**Parameters**:
- **props**: FlowProperties - Fluid properties

**Returns**: float - Prandtl number (dimensionless)

**Formula**: Pr = μ*cp/k

##### grashof_number(self, diameter: float, T_surface: float, T_fluid: float, props: FlowProperties) -> float
**Purpose**: Calculate Grashof number for natural convection

**Parameters**:
- **diameter**: float - Characteristic length (m)
- **T_wall**: float - Wall temperature (K)
- **T_bulk**: float - Bulk fluid temperature (K)
- **props**: FlowProperties - Fluid properties

**Returns**: float - Grashof number (dimensionless)

**Formula**: Gr = g*β*ΔT*L³*ρ²/μ²

##### rayleigh_number(self, grashof: float, prandtl: float) -> float
**Purpose**: Calculate Rayleigh number

**Parameters**:
- **grashof**: float - Grashof number
- **prandtl**: float - Prandtl number

**Returns**: float - Rayleigh number (dimensionless)

**Formula**: Ra = Gr*Pr

##### nusselt_forced_convection_internal(self, reynolds: float, prandtl: float, length_diameter_ratio: Optional[float] = None) -> float
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

##### nusselt_natural_convection_external(self, rayleigh: float, orientation: str, height_diameter_ratio: float = 1.0) -> float
**Purpose**: Calculate natural convection Nusselt number for external surfaces

**Parameters**:
- **rayleigh**: float - Rayleigh number
- **orientation**: str - 'vertical' or 'horizontal'
- **height_diameter_ratio**: float = 1.0 - H/D ratio

**Returns**: float - Nusselt number (dimensionless)

**Notes**: Uses Churchill-Chu correlations

##### nusselt_natural_convection_internal(self, rayleigh: float, orientation: str, length_diameter_ratio: float = 10.0) -> float
**Purpose**: Calculate natural convection Nusselt number for internal flow

**Parameters**:
- **rayleigh**: float - Rayleigh number
- **orientation**: str - 'vertical' or 'horizontal'
- **length_diameter_ratio**: float = 10.0 - L/D ratio

**Returns**: float - Nusselt number (dimensionless)

##### combined_convection_coefficient(self, h_forced: float, h_natural: float) -> float
**Purpose**: Combine forced and natural convection coefficients

**Parameters**:
- **h_forced**: float - Forced convection coefficient (W/m²·K)
- **h_natural**: float - Natural convection coefficient (W/m²·K)

**Returns**: float - Combined coefficient (W/m²·K)

**Formula**: h = (h_forced³ + h_natural³)^(1/3)

##### radiation_heat_transfer_coefficient(self, T_surface: float, T_ambient: float, emissivity: float) -> float
**Purpose**: Calculate linearized radiation heat transfer coefficient

**Parameters**:
- **T_surf**: float - Surface temperature (K)
- **T_surr**: float - Surrounding temperature (K)
- **emissivity**: float - Surface emissivity (0-1)

**Returns**: float - Radiation coefficient (W/m²·K)

**Formula**: h_rad = ε*σ*(T_surf⁴ - T_surr⁴)/(T_surf - T_surr)

##### dew_point_temperature(self, pressure_Pa: float, humidity_ratio: float) -> float
**Purpose**: Calculate dew point temperature

**Parameters**:
- **pressure_Pa**: float - Total pressure in Pa
- **humidity_ratio**: float - kg water / kg dry air

**Returns**: float - Dew point temperature in K

**Method**: Binary search to find temperature where saturation pressure equals vapor pressure

##### condensation_heat_transfer_coefficient(self, props: FlowProperties, wall_temp: float, vapor_temp: float) -> float
**Purpose**: Calculate condensation heat transfer coefficient

**Parameters**:
- **props**: FlowProperties - Fluid properties
- **wall_temp**: float - Wall temperature (K)
- **vapor_temp**: float - Vapor temperature (K)

**Returns**: float - Condensation coefficient (W/m²·K)

**Notes**: Uses Nusselt's laminar film condensation theory

##### pressure_drop(self, props: FlowProperties, diameter: float, length: float, velocity: float, roughness: float = 0.0, fittings_factor: float = 1.0) -> float
**Purpose**: Calculate pressure drop in pipe segment

**Parameters**:
- **props**: FlowProperties - Flow properties
- **diameter**: float - Pipe diameter (m)
- **length**: float - Pipe length (m)
- **velocity**: float - Flow velocity (m/s)
- **roughness**: float = 0.0 - Surface roughness (m)
- **fittings_factor**: float = 1.0 - Factor for additional losses from fittings

**Returns**: float - Pressure drop (Pa)

##### calculate_segment_heat_transfer(self, segment_data: Dict, dt: float) -> Dict
**Purpose**: Main heat transfer solver for a segment with iterative condensation coupling

**Parameters**:
- **segment_data**: Dict - Dictionary containing all segment properties:
  - gas_temperature: float - Gas temperature (K)
  - ambient_temperature: float - Ambient temperature (K)
  - pressure: float - Gas pressure (Pa)
  - velocity: float - Gas velocity (m/s)
  - diameter: float - Inner diameter (m)
  - length: float - Segment length (m)
  - wall_properties: Dict - Material properties
  - wall_type: str - 'single' or 'triple'
  - humidity_ratio: float - Water content (kg/kg)
  - segment_id: int - For tracking condensation state
  - And other optional parameters
- **dt**: float - Time step (s)

**Returns**: Dictionary containing:
- wall_temperature_new: Updated wall temperature (K)
- h_inner: Effective inner convection coefficient (W/m²·K)
- h_outer: Combined outer convection+radiation coefficient (W/m²·K)
- humidity_ratio_new: Updated humidity ratio
- condensation_rate: Condensation rate (kg/s)
- evaporation_rate: Evaporation rate (kg/s)
- film_thickness: Condensate film thickness (m)
- film_mass: Accumulated condensate mass (kg)
- latent_heat_rate: Latent heat effect (W)
- reynolds_number: Flow Reynolds number
- nusselt_forced: Forced convection Nusselt
- nusselt_natural: Natural convection Nusselt
- pressure_drop: Segment pressure drop (Pa)
- buoyancy_pressure: Buoyancy pressure (Pa)
- dew_point: Dew point temperature (K)
- condensation_flag: Whether condensation is occurring
- wall_type: Type of wall construction
- convergence_iterations: Number of iterations to converge
- For triple walls also includes:
  - liner_temperature_new, middle_temperature_new, outer_temperature_new

**Process**:
1. Extract input data from segment_data dictionary
2. Calculate gas properties and dimensionless numbers
3. Iterate to couple heat transfer and condensation:
   - Calculate forced and natural convection coefficients
   - Combine using mixed convection correlation
   - Update condensation state with wall temperature
   - Modify heat transfer for condensate film
   - Calculate latent heat effects
   - Update wall temperatures (single or triple)
   - Check convergence
4. Calculate pressure drop and buoyancy
5. Return comprehensive results dictionary

##### mass_transfer_coefficient(self, props: FlowProperties, diameter: float, velocity: float) -> float
**Purpose**: Calculate mass transfer coefficient using heat-mass analogy

**Parameters**:
- **props**: FlowProperties - Fluid properties
- **diameter**: float - Characteristic length (m)
- **velocity**: float - Flow velocity (m/s)

**Returns**: float - Mass transfer coefficient (m/s)

**Method**: Chilton-Colburn analogy with Lewis number

##### buoyancy_pressure(self, height: float, rho_ambient: float, rho_gas: float) -> float
**Purpose**: Calculate buoyancy pressure difference

**Parameters**:
- **height**: float - Vertical height (m)
- **rho_ambient**: float - Ambient density (kg/m³)
- **rho_gas**: float - Gas density (kg/m³)

**Returns**: float - Buoyancy pressure (Pa)

**Formula**: ΔP = g*h*(ρ_ambient - ρ_gas)

##### air_gap_heat_transfer_coefficient(self, T_hot: float, T_cold: float, gap_width: float, gap_height: float) -> float
**Purpose**: Calculate heat transfer coefficient for air gap between layers

**Parameters**:
- **T_hot**: float - Hot surface temperature (K)
- **T_cold**: float - Cold surface temperature (K)
- **gap_width**: float - Gap width (m)
- **gap_height**: float - Gap height (m)

**Returns**: float - Heat transfer coefficient (W/m²·K)

**Notes**: Includes both conduction and natural convection in enclosed space

##### transient_wall_temperature(self, T_wall_old: float, T_gas: float, T_ambient: float, wall_properties: Dict, geometry: Dict, h_inner: float, h_outer: float, dt: float) -> float
**Purpose**: Calculate wall temperature for next time step

**Parameters**:
- **T_wall_old**: float - Previous wall temperature (K)
- **T_gas**: float - Gas temperature (K)
- **T_ambient**: float - Ambient temperature (K)
- **wall_properties**: Dict - Dict with 'density', 'specific_heat', 'thickness', 'thermal_conductivity'
- **geometry**: Dict - Dict with 'inner_diameter', 'outer_diameter', 'length'
- **h_inner**: float - Inner convection coefficient (W/m²·K)
- **h_outer**: float - Outer convection coefficient (W/m²·K)
- **dt**: float - Time step (s)

**Returns**: float - New wall temperature (K)

**Method**: Stable semi-implicit method to prevent oscillations

##### transient_triple_wall_temperatures(self, T_liner_old: float, T_middle_old: float, T_outer_old: float, T_gas: float, T_ambient: float, wall_properties: Dict, geometry: Dict, h_inner: float, h_outer: float, dt: float) -> Tuple[float, float, float]
**Purpose**: Calculate triple wall temperatures using thermal network

**Parameters**:
- **T_liner_old**: float - Previous liner temperature (K)
- **T_middle_old**: float - Previous middle layer temperature (K)
- **T_outer_old**: float - Previous outer layer temperature (K)
- **T_gas**: float - Gas temperature (K)
- **T_ambient**: float - Ambient temperature (K)
- **wall_properties**: Dict - Material properties for each layer
- **geometry**: Dict - Geometric parameters
- **h_inner**: float - Inner convection coefficient (W/m²·K)
- **h_outer**: float - Outer convection coefficient (W/m²·K)
- **dt**: float - Time step (s)

**Returns**: Tuple of (T_liner_new, T_middle_new, T_outer_new)

**Method**: Thermal resistance network with capacitances


---

## MODULE: coupled_solver.py - Flow-Thermal Coupling
Handles the iterative coupling between flow and heat transfer

### CLASS: ConvergenceMetrics (dataclass)
**Purpose**: Track convergence behavior during iterations

#### Attributes:
- **temperature_residual**: float = 0.0 - Temperature change norm
- **flow_residual**: float = 0.0 - Mass flow change norm  
- **pressure_residual**: float = 0.0 - Pressure change norm
- **iterations**: int = 0 - Number of iterations performed
- **converged**: bool = False - Whether solution converged

### CLASS: CoupledFlowThermalSolver
**Purpose**: Iterative solver for coupled flow-thermal equations

#### Constructor:
```python
__init__(tolerance: float = 1e-3, max_iterations: int = 5,
         relaxation_factor: float = 0.3)
```

**Parameters**:
- **tolerance**: float = 1e-3 - Convergence tolerance for residuals
- **max_iterations**: int = 5 - Maximum coupling iterations per timestep
- **relaxation_factor**: float = 0.3 - Under-relaxation factor for stability

#### Methods:

##### solve_coupled_timestep(self, chimney, step: int, T_inlet: float, dt: float) -> ConvergenceMetrics
**Purpose**: Solve a single timestep with flow-thermal coupling for entire chimney

**Parameters**:
- **chimney**: Chimney object with state and methods
- **step**: int - Current time step index
- **T_inlet**: float - Inlet temperature (K)
- **dt**: float - Time step size (s)

**Returns**: ConvergenceMetrics object with convergence information

**Process**:
1. For first 5 timesteps, use sequential solver for stability
2. Initialize state from previous timestep
3. Main coupling iterations:
   - Update densities from temperatures
   - Calculate buoyancy pressures
   - Solve momentum equation for mass flow
   - Calculate velocities from mass flow
   - Solve energy equation with comprehensive heat transfer
   - Apply adaptive relaxation
   - Check convergence
4. Update chimney state arrays with final values

##### calculate_residuals(self, current: Dict, previous: Dict) -> Tuple[float, float, float]
**Purpose**: Calculate normalized residuals between iterations

**Parameters**:
- **current**: Dict - Current iteration values
- **previous**: Dict - Previous iteration values

**Returns**: Tuple of (temperature_residual, flow_residual, pressure_residual)

##### apply_relaxation(self, new_value: np.ndarray, old_value: np.ndarray) -> np.ndarray
**Purpose**: Apply under-relaxation for stability

**Parameters**:
- **new_value**: np.ndarray - Newly calculated value
- **old_value**: np.ndarray - Previous iteration value

**Returns**: np.ndarray - Relaxed value

**Formula**: relaxed = α * new + (1-α) * old

##### _update_densities(self, chimney, temperatures: np.ndarray) -> np.ndarray
**Purpose**: Update gas densities based on temperatures using ideal gas law

##### _calculate_buoyancy(self, chimney, densities: np.ndarray) -> np.ndarray
**Purpose**: Calculate buoyancy pressure differences for each segment

##### _solve_momentum_improved(self, chimney, pressures, densities, buoyancy, mass_flow_old, dt) -> Tuple[float, np.ndarray]
**Purpose**: Solve momentum equation with improved physics
**Returns**: Tuple of (new_mass_flow, new_pressures)

##### _calculate_velocities(self, chimney, mass_flow: float, densities: np.ndarray) -> np.ndarray
**Purpose**: Calculate velocities from mass flow rate

##### _solve_energy_improved(self, chimney, step, temperatures, velocities, mass_flow, dt) -> np.ndarray
**Purpose**: Solve energy equation using comprehensive heat transfer model
**Note**: Calls chimney.comprehensive_ht.calculate_segment_heat_transfer for each segment

##### _solve_sequential(self, chimney, step: int, T_inlet: float)
**Purpose**: Use sequential (non-iterative) solution for early timesteps

**Note**: Adaptive relaxation ranges from 0.2 to 0.5 based on convergence behavior

---

## MODULE: condensation_physics.py - Phase Change Modeling
Models water vapor condensation and latent heat effects

### CLASS: CondensationState (dataclass)
**Purpose**: Track condensation state at a point

#### Attributes:
- **film_thickness**: float = 0.0 - Liquid film thickness (m)
- **film_mass**: float = 0.0 - Accumulated condensate (kg)
- **condensation_rate**: float = 0.0 - Mass condensation rate (kg/s)
- **evaporation_rate**: float = 0.0 - Mass evaporation rate (kg/s)
- **net_rate**: float = 0.0 - Net mass transfer rate (kg/s)
- **latent_heat_rate**: float = 0.0 - Latent heat release (W)

### CLASS: CondensationPhysics
**Purpose**: Handle condensation calculations and thermodynamics

#### Constants:
- **R_UNIVERSAL**: float = 8314.0 - Universal gas constant J/(kmol·K)
- **R_WATER**: float = 461.5 - Gas constant for water vapor (J/kg·K)
- **R_AIR**: float = 287.0 - Gas constant for dry air (J/kg·K)
- **MW_WATER**: float = 18.015 - Molecular weight of water (kg/kmol)
- **MW_AIR**: float = 28.97 - Molecular weight of dry air (kg/kmol)
- **H_FG_REF**: float = 2.501e6 - Latent heat at 0°C (J/kg)
- **CP_WATER_VAPOR**: float = 1859.0 - Specific heat of water vapor (J/kg·K)
- **CP_LIQUID_WATER**: float = 4186.0 - Specific heat of liquid water (J/kg·K)
- **RHO_LIQUID_WATER**: float = 1000.0 - Density of liquid water (kg/m³)

#### Methods:

##### saturation_pressure(self, T_K: float) -> float
**Purpose**: Calculate water saturation pressure

**Parameters**:
- **T_K**: float - Temperature (K)

**Returns**: float - Saturation pressure (Pa)

**Method**: 
- Below -50°C: Returns near zero
- -50°C to 0°C: Tetens formula for ice
- 0°C to 100°C: Buck equation for water
- Above 100°C: Antoine equation for steam

##### latent_heat(self, T_K: float) -> float
**Purpose**: Calculate latent heat of vaporization

**Parameters**:
- **T_K**: float - Temperature (K)

**Returns**: float - Latent heat (J/kg)

**Formula**: Watson correlation with temperature dependence

##### humidity_ratio_to_mole_fraction(self, W: float) -> float
**Purpose**: Convert humidity ratio to mole fraction

**Parameters**:
- **W**: float - Humidity ratio (kg water / kg dry air)

**Returns**: float - Mole fraction of water vapor

**Formula**: x = W * MW_air / (W * MW_air + MW_water)

##### mole_fraction_to_humidity_ratio(self, x_water: float) -> float
**Purpose**: Convert mole fraction to humidity ratio

**Parameters**:
- **x_water**: float - Mole fraction of water vapor

**Returns**: float - Humidity ratio (kg water/kg dry air)

**Formula**: W = x * MW_water / ((1 - x) * MW_air)

##### mass_transfer_coefficient(self, h_heat: float, T_gas: float, P_total: float) -> float
**Purpose**: Calculate mass transfer coefficient using heat-mass analogy

**Parameters**:
- **h_heat**: float - Heat transfer coefficient (W/m²·K)
- **T_gas**: float - Gas temperature (K)
- **P_total**: float - Total pressure (Pa)

**Returns**: float - Mass transfer coefficient (kg/m²·s)

**Method**: Chilton-Colburn analogy with Lewis number ≈ 1

##### condensation_rate(self, T_gas: float, T_wall: float, P_total: float, W_gas: float, h_heat: float, area: float) -> Tuple[float, float]
**Purpose**: Calculate condensation rate using physics-based approach

**Parameters**:
- **T_gas**: float - Gas temperature (K)
- **T_wall**: float - Wall surface temperature (K)
- **P_total**: float - Total pressure (Pa)
- **W_gas**: float - Gas humidity ratio (kg water / kg dry air)
- **h_heat**: float - Heat transfer coefficient (W/m²·K)
- **area**: float - Surface area (m²)

**Returns**: Tuple of (condensation_rate kg/s, latent_heat_rate W)

##### evaporation_rate(self, T_wall: float, T_gas: float, P_total: float, W_gas: float, h_heat: float, area: float, film_mass: float) -> Tuple[float, float]
**Purpose**: Calculate evaporation rate from existing condensate film

**Parameters**:
- **T_wall**: float - Wall surface temperature (K)
- **T_gas**: float - Gas temperature (K)
- **P_total**: float - Total pressure (Pa)
- **W_gas**: float - Gas humidity ratio (kg water / kg dry air)
- **h_heat**: float - Heat transfer coefficient (W/m²·K)
- **area**: float - Surface area (m²)
- **film_mass**: float - Existing film mass (kg)

**Returns**: Tuple of (evaporation_rate kg/s, latent_heat_rate W)

##### film_heat_transfer_modification(self, h_base: float, film_thickness: float, k_liquid: float = 0.6) -> float
**Purpose**: Modify heat transfer coefficient due to condensate film

**Parameters**:
- **h_base**: float - Base heat transfer coefficient without film (W/m²·K)
- **film_thickness**: float - Condensate film thickness (m)
- **k_liquid**: float = 0.6 - Thermal conductivity of liquid water (W/m·K)

**Returns**: float - Modified heat transfer coefficient (W/m²·K)

##### update_condensation_state(self, state: CondensationState, T_gas: float, T_wall: float, P_total: float, W_gas: float, h_heat: float, area: float, dt: float) -> CondensationState
**Purpose**: Update condensation state with proper mass balance

**Parameters**:
- **state**: CondensationState - Current condensation state
- **T_gas**: float - Gas temperature (K)
- **T_wall**: float - Wall surface temperature (K)
- **P_total**: float - Total pressure (Pa)
- **W_gas**: float - Gas humidity ratio (kg water / kg dry air)
- **h_heat**: float - Heat transfer coefficient (W/m²·K)
- **area**: float - Surface area (m²)
- **dt**: float - Time step (s)

**Returns**: Updated CondensationState

##### gas_humidity_change(self, W_gas: float, condensation_net: float, mass_flow: float, volume: float, density: float, dt: float) -> float
**Purpose**: Calculate change in gas humidity due to condensation/evaporation

**Parameters**:
- **W_gas**: float - Current humidity ratio
- **condensation_net**: float - Net condensation rate (kg/s)
- **mass_flow**: float - Gas mass flow rate (kg/s)
- **volume**: float - Control volume (m³)
- **density**: float - Gas density (kg/m³)
- **dt**: float - Time step (s)

**Returns**: float - Updated humidity ratio

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

**Note**: This solver models triple walls with 3 temperature nodes (liner, middle, outer) rather than 5 nodes as mentioned in code comments.

#### Constants:
- **STEFAN_BOLTZMANN**: float = 5.67e-8 - Stefan-Boltzmann constant (W/m²·K⁴)

#### Constructor:
```python
__init__(self)
```
**Note**: Simple initialization with no parameters

#### Methods:

##### calculate_view_factors(self, r_inner: float, r_outer: float) -> Tuple[float, float]
**Purpose**: Calculate radiation view factors for concentric cylinders

**Parameters**:
- **r_inner**: float - Inner radius (m)
- **r_outer**: float - Outer radius (m)

**Returns**: Tuple of (F_inner_to_outer, F_outer_to_inner)

**Notes**:
- F_inner_to_outer = 1.0 (all radiation from inner hits outer)
- F_outer_to_inner = r_inner/r_outer (reciprocity relation)

##### radiation_resistance(self, r_inner: float, r_outer: float, emiss_inner: float, emiss_outer: float, length: float) -> float
**Purpose**: Calculate radiation resistance between concentric cylinders

**Parameters**:
- **r_inner**: float - Inner radius (m)
- **r_outer**: float - Outer radius (m)
- **emiss_inner**: float - Inner surface emissivity (0-1)
- **emiss_outer**: float - Outer surface emissivity (0-1)
- **length**: float - Cylinder length (m)

**Returns**: float - Radiation resistance (K/W)

**Formula**: Three resistances in series:
- R1 = (1-ε_inner)/(ε_inner * A_inner) - Inner emissivity resistance
- R2 = 1/A_inner - Geometric/space resistance
- R3 = (1-ε_outer)/(ε_outer * A_outer) - Outer emissivity resistance
- R_rad = R1 + R2 + R3

##### scale_system(self, K_matrix: np.ndarray, C_matrix: np.ndarray, Q_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
**Purpose**: Scale system matrices to improve conditioning

**Parameters**:
- **K_matrix**: np.ndarray - Conductance matrix (W/K)
- **C_matrix**: np.ndarray - Capacitance matrix (J/K)
- **Q_vector**: np.ndarray - Heat source vector (W)

**Returns**: Tuple of (scaled_K, scaled_C, scaled_Q, scaling_factors)

**Method**: 
- Calculate scale_factors[i] = sqrt(K[i,i] * C[i,i]) for valid diagonal elements
- Create diagonal scaling matrix S = diag(1/scale_factors)
- Return S @ K @ S_inv, S @ C @ S_inv, S @ Q, scale_factors

##### solve_triple_wall_stable(self, T_gas: float, T_ambient: float, h_inner: float, h_outer: float, T_liner_old: float, T_middle_old: float, T_outer_old: float, geometry: Dict, properties: Dict, dt: float) -> Tuple[float, float, float]
**Purpose**: Solve triple-wall heat transfer with implicit Euler method

**Parameters**:
- **T_gas**: float - Gas temperature (K)
- **T_ambient**: float - Ambient temperature (K)
- **h_inner**: float - Inner convection coefficient (W/m²·K)
- **h_outer**: float - Outer convection coefficient (W/m²·K)
- **T_liner_old**: float - Previous liner temperature (K)
- **T_middle_old**: float - Previous middle layer temperature (K)
- **T_outer_old**: float - Previous outer layer temperature (K)
- **geometry**: Dict - Must contain:
  - inner_diameter: Inner diameter (m)
  - length: Segment length (m)
- **properties**: Dict - Must contain:
  - liner: WallLayerProperties object
  - middle: WallLayerProperties object
  - outer: WallLayerProperties object
- **dt**: float - Time step (s)

**Returns**: Tuple of (T_liner_new, T_middle_new, T_outer_new)

**Process**:
1. Extract and validate geometry (calculates radii from thicknesses)
2. Calculate surface areas for heat transfer
3. Compute conduction resistances using cylindrical formula: R = ln(r_o/r_i)/(2πkL)
4. Calculate convection resistances with bounds (h_min = 1.0 W/m²·K)
5. Calculate radiation resistances between layers (if emissivities provided)
6. Calculate thermal capacitances for each layer
7. Build 3x3 conductance matrix K with conductances G = 1/(R + eps)
8. Build 3x3 diagonal capacitance matrix C
9. Create source vector Q with boundary conditions
10. Scale system for better conditioning
11. Solve implicit system: (C/dt + K)*T_new = C/dt*T_old + Q
12. If matrix is singular, fall back to explicit update
13. Apply physical bounds (T_min = 173.15K, T_max = 1273.15K)

**Error Handling**:
- Extensive validation of geometry and properties
- Detailed diagnostics printed if matrix contains inf/nan
- Fallback to explicit method if matrix is singular
- Physical bounds applied to prevent unrealistic temperatures

##### linearize_radiation(self, T_ref: float, emissivity: float) -> float
**Purpose**: Linearize radiation heat transfer coefficient

**Parameters**:
- **T_ref**: float - Reference temperature for linearization (K)
- **emissivity**: float - Surface emissivity (0-1)

**Returns**: float - Linearized radiation coefficient (W/m²·K)

**Formula**: h_rad = 4*ε*σ*T_ref³

**Note**: This method exists but is NOT USED in the current implementation

---

## MODULE: adaptive_triple_wall.py - Adaptive Wall Solver
Adaptive time-stepping solver for stiff triple-wall problems

### CLASS: AdaptiveTripleWallSolver
**Purpose**: Solve triple-wall heat transfer with adaptive sub-stepping to avoid tiny global timesteps

#### Constructor:
```python
__init__(self)
```
**Note**: Initializes with STEFAN_BOLTZMANN constant = 5.67e-8

#### Attributes:
- **STEFAN_BOLTZMANN**: float = 5.67e-8 - Stefan-Boltzmann constant (W/m²·K⁴)

#### Methods:

##### solve_with_adaptive_dt(self, T_gas: float, T_ambient: float, h_inner: float, h_outer: float, T_liner_old: float, T_middle_old: float, T_outer_old: float, geometry: Dict, properties: Dict, dt_global: float) -> Tuple[float, float, float]
**Purpose**: Solve triple wall with adaptive sub-stepping for numerical stability

**Parameters**:
- **T_gas**: float - Gas temperature (K)
- **T_ambient**: float - Ambient temperature (K)
- **h_inner**: float - Inner convection coefficient (W/m²·K)
- **h_outer**: float - Outer convection coefficient (W/m²·K)
- **T_liner_old**: float - Previous liner temperature (K)
- **T_middle_old**: float - Previous middle layer temperature (K)
- **T_outer_old**: float - Previous outer layer temperature (K)
- **geometry**: Dict - Must contain:
  - inner_diameter: Inner diameter (m)
  - length: Segment length (m)
- **properties**: Dict - Must contain:
  - liner: Properties object with thickness, density, specific_heat, thermal_conductivity
  - middle: Properties object with thickness, density, specific_heat, thermal_conductivity
  - outer: Properties object with thickness, density, specific_heat, thermal_conductivity
- **dt_global**: float - Global simulation timestep (can be large)

**Returns**: Tuple of (T_liner_new, T_middle_new, T_outer_new)

**Process**:
1. Calculate geometry (radii and volumes for each layer)
2. Calculate thermal masses: C = ρ * cp * V
3. Estimate thermal time constants: τ = ρ * cp * thickness² / k
4. Determine required timestep: dt_required = 0.1 * min(τ₁, τ₂, τ₃)
5. If dt_global <= dt_required:
   - Use single step with dt_global
6. Else:
   - n_substeps = ceil(dt_global/dt_required)
   - dt_sub = dt_global/n_substeps
   - Loop through n_substeps calling _solve_single_step

##### _solve_single_step(self, T_gas: float, T_ambient: float, h_inner: float, h_outer: float, T_liner_old: float, T_middle_old: float, T_outer_old: float, geometry: Dict, properties: Dict, dt: float) -> Tuple[float, float, float]
**Purpose**: Solve single timestep using semi-implicit method

**Parameters**: Same as solve_with_adaptive_dt except dt is a single timestep

**Returns**: Tuple of (T_liner_new, T_middle_new, T_outer_new)

**Method**: Semi-implicit energy balance
1. Calculate geometry (radii, areas, volumes)
2. Calculate thermal masses and conduction resistances
3. Update temperatures sequentially:
   - **Liner**: Q_in = h_inner * A_gas * (T_gas - T_liner_old), Q_out to middle layer
   - **Middle**: Q_in from liner using T_liner_new, Q_out to outer layer  
   - **Outer**: Q_in from middle using T_middle_new, Q_out = h_outer * A_3 * (T_outer - T_ambient)
4. Apply physical bounds:
   - T_liner ∈ [T_ambient, T_gas]
   - T_middle ∈ [T_ambient, T_liner_new]
   - T_outer ∈ [T_ambient, T_middle_new]

**Simplifications**:
- No radiation between layers (unlike StableTripleWallSolver)
- Uses cylindrical conduction resistance: R = ln(r_o/r_i)/(2πkL)
- Semi-implicit method (forward differences for upstream temperatures)

**Note**: This solver is actually USED by comprehensive_heat_transfer.py as an alternative to StableTripleWallSolver

---

## MODULE: energy_diagnostics.py - Conservation Checks
Monitors energy conservation throughout simulation

### CLASS: EnergyDiagnostics
**Purpose**: Check energy conservation and physical bounds at each timestep

#### Constructor:
```python
__init__(self)
```
**Initializes**: Empty lists for warnings and errors

#### Attributes:
- **warnings**: List[str] - Accumulated warning messages
- **errors**: List[str] - Accumulated error messages

#### Methods:

##### check_segment_energy_balance(self, segment_data: Dict[str, Any], dt: float, results: Dict[str, Any]) -> Dict[str, float]
**Purpose**: Verify energy changes respect physical limits

**Parameters**:
- **segment_data**: Dict - Must contain:
  - gas_temperature: Gas temperature (K)
  - ambient_temperature: Ambient temperature (K)
  - diameter: Inner diameter (m)
  - length: Segment length (m)
  - pressure: Gas pressure (Pa)
- **dt**: float - Time step (s)
- **results**: Dict - Heat transfer results containing:
  - h_inner: Inner heat transfer coefficient
  - wall_temperature_new: Updated wall temperature (optional)

**Returns**: Dictionary with diagnostic values:
- available_energy_J: Total thermal energy above ambient (J)
- max_heat_loss_W: Maximum possible heat loss rate (W)
- max_cooling_K: Maximum temperature drop possible (K)
- time_to_ambient_s: Time to cool to ambient (s)

**Checks performed**:
1. Calculates available thermal energy in gas
2. Checks if timestep > 0.1 * time_to_ambient (warning)
3. Checks if T_gas < T_ambient - 0.1 (error)
4. Checks if T_wall < T_ambient - 0.1 (error)

**Note**: Uses approximate cp_gas = 1000 J/(kg·K)

##### check_condensation_physics(self, T_gas: float, T_wall: float, T_ambient: float, condensation_rate: float) -> None
**Purpose**: Verify condensation only occurs under correct conditions

**Parameters**:
- **T_gas**: float - Gas temperature (K)
- **T_wall**: float - Wall temperature (K)
- **T_ambient**: float - Ambient temperature (K)
- **condensation_rate**: float - Condensation rate (kg/s)

**Side Effects**: Adds to errors/warnings lists

**Checks**:
- Error if condensation_rate > 0 and T_wall >= T_gas
- Warning if T_wall < T_ambient and condensation_rate > 0

##### get_report(self) -> str
**Purpose**: Generate formatted diagnostic report

**Returns**: String containing:
- "ERRORS:" section if errors exist
- "WARNINGS:" section if warnings exist
- "Energy conservation checks passed" if no issues

##### clear(self) -> None
**Purpose**: Clear diagnostics for next timestep

**Side Effects**: Resets warnings and errors lists to empty

**IMPORTANT NOTE**: This class is instantiated in comprehensive_heat_transfer.py but its methods are NEVER CALLED in the current implementation. It appears to be diagnostic code that was prepared but not integrated into the main simulation loop.

---

## MODULE: early_diagnostics.py - Startup Diagnostics
Monitors simulation startup for numerical issues

### CLASS: EarlyTimeDiagnostics
**Purpose**: Monitor early timesteps to catch instabilities before they grow

#### Constructor:
```python
__init__(self, ambient_temp_K: float = 293.15)
```

**Parameters**:
- **ambient_temp_K**: float = 293.15 - Ambient temperature in Kelvin

**Initializes**:
- self.ambient_temp: Reference ambient temperature
- self.errors: Empty list for error messages
- self.warnings: Empty list for warning messages

#### Attributes:
- **ambient_temp**: float - Ambient temperature (K)
- **errors**: List[str] - Current error messages
- **warnings**: List[str] - Current warning messages

#### Methods:

##### check_timestep(self, step: int, time: float, T_gas: np.ndarray, T_wall: np.ndarray, T_liner: np.ndarray = None, T_middle: np.ndarray = None, T_outer: np.ndarray = None, mass_flow: float = 0, dt: float = 0) -> None
**Purpose**: Check for unphysical behavior at each timestep

**Parameters**:
- **step**: int - Current time step number
- **time**: float - Current simulation time (s)
- **T_gas**: np.ndarray - Gas temperatures for all segments (K)
- **T_wall**: np.ndarray - Wall temperatures for all segments (K)
- **T_liner**: np.ndarray = None - Triple wall liner temperatures (K)
- **T_middle**: np.ndarray = None - Triple wall middle temperatures (K)
- **T_outer**: np.ndarray = None - Triple wall outer temperatures (K)
- **mass_flow**: float = 0 - Mass flow rate (kg/s)
- **dt**: float = 0 - Time step size (s)

**Side Effects**:
- Clears previous errors/warnings
- Populates errors/warnings lists with new issues
- Prints diagnostics to console for step < 10

**Checks Performed**:
1. Gas temperature < ambient - 1.0 K (error with segment location)
2. Gas temperature < 0 K (critical error)
3. Wall temperature < ambient - 1.0 K (error with segment location)
4. Wall temperature < 0 K (critical error)
5. Liner temperature < ambient - 1.0 K (if triple wall)
6. Mass flow < 0 (negative flow error)

**Note**: Cooling rate check is commented but not implemented

##### get_report(self) -> str
**Purpose**: Generate formatted diagnostic report

**Returns**: String containing:
- "No issues detected" if no errors/warnings
- "ERRORS:" section with error list
- "WARNINGS:" section with warning list

**IMPORTANT NOTE**: This class IS ACTUALLY USED in the simulation. It's instantiated in chimney.py and check_timestep is called every timestep. Unlike EnergyDiagnostics, this diagnostic code is integrated into the main simulation loop.

**Methods NOT IMPLEMENTED**:
- check_temperature_bounds() - Does not exist
- check_convergence_quality() - Does not exist
- diagnose_instability() - Does not exist

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

**Constructor**:
```python
__init__(self, chimney)
```

**Attributes**:
- **chimney**: Reference to chimney object
- **g**: float = 9.81 - Gravitational acceleration (m/s²)

### Methods

##### calculate_draft(self) -> DraftResults
**Purpose**: Calculate draft pressure for all timesteps

**Returns**: DraftResults object with complete draft analysis

**Calculations**:
1. Iterates through all timesteps and segments
2. Calculates segment-by-segment buoyancy: ΔP = g·h·(ρ_ambient - ρ_gas)
3. Calculates friction losses using Reynolds number and friction factor
4. Adds fitting losses (K=0.5) if has_fittings is True
5. Available draft = total draft - losses
6. Theoretical velocity from Bernoulli: v = √(2·ΔP/ρ)

**Note**: Skips pressure loss calculation for first timestep (no flow yet)

##### _friction_factor(self, Re: float, relative_roughness: float) -> float
**Purpose**: Calculate Darcy friction factor

**Parameters**:
- **Re**: float - Reynolds number
- **relative_roughness**: float - ε/D ratio

**Returns**: float - Friction factor

**Correlations**:
- Laminar (Re < 2300): f = 64/max(Re, 1)
- Smooth turbulent (ε/D < 1e-6): f = 0.316/Re^0.25 (Blasius)
- Rough turbulent: f = 0.25/[log10(ε/3.7D + 5.74/Re^0.9)]² (Colebrook approximation)

##### plot_draft_analysis(self, results: DraftResults, save_path: Optional[str] = None)
**Purpose**: Create comprehensive 2x2 draft analysis plots

**Parameters**:
- **results**: DraftResults - Results from calculate_draft()
- **save_path**: Optional[str] = None - Path to save figure

**Creates 4 subplots**:
1. **Top Left**: Total draft, available draft, and losses vs time
2. **Top Right**: Draft contribution by segment (bar chart at final time)
3. **Bottom Left**: Mass flow rate and draft velocity vs time (dual axis)
4. **Bottom Right**: Draft efficiency (available/total × 100) vs time

**Side Effects**: Displays matplotlib figure; saves if path provided

##### print_draft_summary(self, results: DraftResults)
**Purpose**: Print summary of draft analysis results to console

**Parameters**:
- **results**: DraftResults - Results from calculate_draft()

**Prints**:
- Chimney configuration (height, segments)
- Steady state results at final time:
  - Total draft pressure
  - Pressure losses
  - Available draft
  - Draft efficiency percentage

**IMPORTANT NOTE**: This class IS ACTUALLY USED in the simulation. It's instantiated in chimney.py at the end of run_simulation(), calculate_draft() is called, and print_draft_summary() displays results. The plot_draft_analysis() method exists but is NOT CALLED in the main simulation - it's available for post-processing visualization.

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

**CRITICAL NOTE**: THE ENTIRE MODULE IS CURRENTLY COMMENTED OUT - All code in this file is commented and non-functional.

### Status
- **File exists but is entirely commented out**
- Functions are imported by chimney.py but will fail if called
- The Chimney class has a wrapper method plot_chimney_snapshot() that references this module

### Commented Functions (NOT AVAILABLE)

#### plot_chimney_geometry() - COMMENTED OUT
Would create animated visualization of chimney geometry with temperature evolution if uncommented.

#### plot_chimney_snapshot() - COMMENTED OUT  
Would plot single timestep of chimney geometry with temperatures if uncommented.

### Original Intent (when uncommented)
The module was designed to:
- Create animated GIFs of temperature evolution
- Generate static snapshots at specific timesteps
- Use Rectangle patches for geometry visualization
- Apply 'hot' colormap for gas temperatures
- Apply 'viridis' colormap for wall temperatures
- Support both vertical and horizontal orientations
- Scale wall thickness for visibility

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