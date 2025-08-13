# Chimney/Flue Heat Transfer Simulation Package

## Overview

This package provides a comprehensive simulation framework for analyzing transient heat transfer and fluid dynamics in chimney/flue systems. It solves the coupled momentum-energy equations to predict temperature distributions, mass flow rates, pressure drops, and condensation behavior in vertical chimney configurations. The simulation handles both single-wall and triple-wall (insulated) chimney designs with full consideration of conduction, convection, radiation, and phase change phenomena.

## Physical Models and Governing Equations

### 1. Conservation Equations

#### Mass Conservation (Continuity)
For steady flow through each segment:
```
ṁ_in = ṁ_out = ρ·A·v = constant
```
Where:
- ṁ = mass flow rate (kg/s)
- ρ = gas density (kg/m³)
- A = cross-sectional area (m²)
- v = velocity (m/s)

#### Momentum Conservation (Draft Equation)
The pressure balance for buoyancy-driven flow:
```
ΔP_buoyancy - ΔP_friction - ΔP_minor = 0
```

Where:
- ΔP_buoyancy = g·H·(ρ_ambient - ρ_gas) = driving force
- ΔP_friction = f·(L/D)·(ρv²/2) = wall friction losses
- ΔP_minor = K·(ρv²/2) = entry/exit losses

#### Energy Conservation
For gas phase:
```
ρ·cp·∂T/∂t + ρ·cp·v·∂T/∂z = -q_wall/A_gas
```

For wall:
```
ρ_w·cp_w·∂T_w/∂t = q_in - q_out
```

### 2. Heat Transfer Mechanisms

#### Convection Correlations

**Forced Convection (Internal Flow)**
- Laminar (Re < 2300): Nu = 3.66 for fully developed flow
- Transition (2300 < Re < 4000): Linear interpolation
- Turbulent (Re > 4000): Nu = 0.023·Re^0.8·Pr^0.4 (Dittus-Boelter)

**Natural Convection (Internal Flow)**
For vertical channels with Rayleigh number Ra = Gr·Pr:
```
Nu = C·Ra^n
```
Where C and n depend on flow regime and geometry

**Mixed Convection**
Richardson number determines regime:
```
Ri = Gr/Re² = (buoyancy forces)/(inertial forces)
```
Blending function:
```
h_mixed = h_forced·(1 - tanh(Ri)) + h_natural·tanh(Ri)
```

**External Natural Convection**
Churchill-Chu correlation for vertical cylinders:
```
Nu = {0.825 + 0.387·Ra^(1/6)/[1 + (0.492/Pr)^(9/16)]^(8/27)}²
```

#### Conduction

**Single Wall (Cylindrical)**
```
R_cond = ln(r_o/r_i)/(2π·k·L)
Q = 2π·k·L·(T_i - T_o)/ln(r_o/r_i)
```

**Triple Wall (Resistance Network)**
Total resistance:
```
R_total = R_conv,in + R_cond,1 + R_gap,12 + R_cond,2 + R_gap,23 + R_cond,3 + R_conv,out
```

Where gap resistances include both conduction and radiation.

#### Radiation

**Between Concentric Cylinders**
```
Q_rad = σ·A_i·(T_i⁴ - T_o⁴)/[1/ε_i + (A_i/A_o)·(1/ε_o - 1)]
```

Linearized for numerical stability:
```
h_rad = 4·ε_eff·σ·T_mean³
```

### 3. Fluid Properties

#### Gas Density (Ideal Gas Law)
```
ρ = P/(R·T)
```
Where R = 287.05 J/(kg·K) for dry air

#### Dynamic Viscosity (Sutherland's Law)
```
μ = μ_ref·(T/T_ref)^(3/2)·(T_ref + S)/(T + S)
```
With μ_ref = 1.716×10⁻⁵ Pa·s, T_ref = 273.15 K, S = 110.4 K

#### Thermal Conductivity
```
k = 0.0241·(T/273.15)^0.9 W/(m·K)
```

#### Specific Heat
```
cp = 1000 + 0.17·(T - 273.15) J/(kg·K)
```

### 4. Dimensionless Numbers

#### Reynolds Number
```
Re = ρ·v·D/μ = (inertial forces)/(viscous forces)
```

#### Prandtl Number
```
Pr = μ·cp/k = (momentum diffusivity)/(thermal diffusivity)
```

#### Grashof Number
```
Gr = g·β·ΔT·L³·ρ²/μ² = (buoyancy forces)/(viscous forces)
```

#### Rayleigh Number
```
Ra = Gr·Pr = g·β·ΔT·L³/(ν·α)
```

### 5. Condensation Physics

#### Saturation Pressure
Antoine equation:
```
log₁₀(P_sat) = A - B/(C + T)
```

#### Humidity Ratio
```
ω = 0.622·P_vapor/(P_total - P_vapor)
```

#### Dew Point Temperature
Solved iteratively from:
```
P_vapor = P_sat(T_dew)
```

#### Film Condensation (Nusselt Theory)
```
h_cond = 0.943·[g·ρ_l·(ρ_l - ρ_v)·h_fg·k_l³/(μ_l·L·ΔT)]^0.25
```

#### Mass Transfer
Using Chilton-Colburn analogy:
```
Sh = Nu·(Sc/Pr)^(1/3)
ṁ_cond = h_m·A·(ω_bulk - ω_wall)
```

### 6. Friction Factor Correlations

#### Smooth Pipes
- Laminar: f = 64/Re
- Turbulent: Colebrook-White equation (solved iteratively)
```
1/√f = -2·log₁₀(ε/D/3.7 + 2.51/(Re·√f))
```

## Numerical Methods

### 1. Spatial Discretization

The chimney is divided into N segments of height Δz. Each segment uses:
- Upwind differencing for convection terms
- Central differencing for conduction
- Control volume energy balance

### 2. Time Integration

#### Gas Temperature (Explicit/Semi-implicit)
```
T_gas^(n+1) = T_gas^n + Δt·[-v·(T_out - T_in)/Δz - Q_wall/(ρ·cp·V)]
```

#### Wall Temperature (Semi-Implicit)
For single wall:
```
C_w·(T_w^(n+1) - T_w^n)/Δt = Q_in^(n+1) - Q_out^(n+1)
```

For triple wall, solve matrix system:
```
[C/Δt + K]·T^(n+1) = [C/Δt]·T^n + Q
```

### 3. Coupled Solution Algorithm

1. **Initialize**: Set T_gas = T_inlet, T_wall = T_ambient
2. **For each time step**:
   a. Calculate gas properties from current temperatures
   b. **Iterate** (typically 3-5 iterations):
      - Solve momentum equation for ṁ
      - Update velocities v = ṁ/(ρ·A)
      - Calculate heat transfer coefficients
      - Solve energy equation for T_gas
      - Solve wall heat conduction for T_wall
      - Check convergence |T_new - T_old| < ε
   c. Apply relaxation if needed: T = α·T_new + (1-α)·T_old
   d. Check energy conservation

### 4. Numerical Stability

#### CFL Condition (Convection)
```
Δt < CFL·Δz/v_max
```
Typically CFL = 0.5

#### Fourier Number (Conduction)
```
Fo = α·Δt/Δx² < 0.5
```

#### Adaptive Time-Stepping
For triple walls with disparate time scales:
```
τ_metal ~ ρ·cp·δ²/k ~ 1-10 s
τ_insulation ~ ρ·cp·δ²/k ~ 100-1000 s
```
Sub-stepping used when Δt_global > 0.1·τ_min

### 5. Matrix Conditioning

For triple-wall systems, the solver monitors condition number:
- Warns when condition number > 1e10
- Provides diagnostics showing capacitances and conductances
- Suggests time step adjustments when needed

## Validation Approach

### 1. Conservation Checks
- Mass: Σ(ṁ_in) = Σ(ṁ_out)
- Energy: E_stored + E_lost = E_input ± 1%

### 2. Physical Bounds
- T_ambient ≤ T_wall ≤ T_gas (for heating)
- Monotonic temperature profiles
- Positive mass flow rates

### 3. Benchmark Cases
- Steady-state: Compare with analytical solutions
- Natural convection: Compare with established correlations
- Fire curve response: Compare with experimental data

## Implementation Features

### 1. Boundary Conditions

**Inlet (z = 0)**
- T_gas specified (ISO 834 fire curve or custom)
- P = P_ambient + ρ·g·H

**Outlet (z = H)**
- ∂T/∂z = 0 (zero gradient)
- P = P_ambient

**Wall External Surface**
- Convection: q = h·(T_wall - T_ambient)
- Radiation: q = ε·σ·(T_wall⁴ - T_ambient⁴)

### 2. Special Capabilities

**ISO 834 Fire Curve**
```
T_fire = 20 + 345·log₁₀(8·t + 1) °C
```
Where t is time in minutes.

**Condensation Tracking**
- Monitors dew point and wall temperature
- Calculates condensation rate and location
- Tracks accumulated condensate mass

**Energy Diagnostics**
- Real-time energy balance monitoring
- Detailed breakdown by heat transfer mode
- Automatic warnings for conservation violations

### 3. Solver Selection

**Single Wall**: Direct solution of heat balance

**Triple Wall**: 
- Stable solver: Full implicit with radiation
- Adaptive solver: Sub-stepping for thin liners
- Automatic selection based on:
  - Liner thickness < 1mm → Adaptive
  - Δt > 10·τ_liner → Adaptive
  - Otherwise → Stable

## Applications

1. **Design Analysis**
   - Sizing for required draft
   - Insulation optimization
   - Material selection

2. **Safety Studies**
   - Fire response analysis
   - Condensation prevention
   - Thermal stress evaluation

3. **Performance Optimization**
   - Startup time reduction
   - Energy efficiency improvement
   - Operating envelope definition

## Key Assumptions and Limitations

1. **Flow Assumptions**
   - Quasi-steady flow (mass storage neglected)
   - One-dimensional flow
   - Ideal gas behavior

2. **Heat Transfer Assumptions**
   - Gray surfaces for radiation
   - Uniform properties in each segment
   - Negligible axial conduction in gas

3. **Geometric Limitations**
   - Vertical or near-vertical orientation
   - Circular cross-sections
   - No branching or merging

4. **Validity Ranges**
   - Temperature: 200 K < T < 2000 K
   - Pressure: Near atmospheric
   - Reynolds: 10 < Re < 10⁶
   - Prandtl: 0.5 < Pr < 1.0 (gases)