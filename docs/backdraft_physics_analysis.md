# Backdraft Physics Analysis

**Branch:** `investigation/backdraft-physics` (from `paper/results`)

**Goal:** Determine exactly why simulated S1 temperature remains cold during
backdraft in the 0227 depressurization test, and what physical mechanism is
missing or misrepresented. No solver modifications — analysis only.

---

## Section A — Gas Energy Equation During Reverse Flow

### A1. Discrete gas energy equation

The gas energy solver is `_solve_energy_improved` in
`flued_module/coupled_solver.py` (lines 657–854).

For each interior segment (not inlet boundary, `|mdot| > 1e-6`), the default
`"current"` model computes:

```
q_loss_to_walls = h_inner * A_wall * (T_gas[seg] - T_wall)       [line 816]
residence_time  = gas_mass / |mdot|                               [line 817]
q_latent        = condensation latent heat rate                   [line 818]
q_total_loss    = q_loss_to_walls + q_latent                      [line 819]

T_steady = T_upstream - q_total_loss * residence_time / (gas_mass * cp)  [line 821]
```

Bounded to `[min(T_upstream, T_wall) - 1, max(T_upstream, T_wall) + 1]`
(lines 825–828).

Then relaxed toward T_steady with a thermal time constant:

```
tau_thermal = gas_mass * cp / (|mdot| * cp + h_inner * A_wall)   [line 830]
alpha       = dt / (tau_thermal + dt)                             [line 831]
T_new[seg]  = T_old[seg] + alpha * (T_steady - T_old[seg])       [line 832]
```

**Terms present:**
- Advection: via `T_upstream` (upwind temperature)
- Wall convection: via `h_inner * A_wall * (T_gas - T_wall)` (includes
  forced, natural, and linearized radiation — see Section B)
- Condensation latent heat: via `q_latent`
- Conduction between gas segments: NOT present (no axial diffusion term)
- Direct radiation to gas (volumetric absorption): NOT present

### A2. Upstream temperature selection when mdot < 0

Lines 739–749:

```python
if mass_flow >= 0:
    if seg == 0:  T_upstream = temperatures[0]   # self-reference at inlet
    else:         T_upstream = temperatures[seg-1]
else:
    if seg == chimney.total_segments - 1:
        T_upstream = part.T_ambient_K             # top = cold air from outside
    else:
        T_upstream = temperatures[seg+1]          # air flows downward from above
```

During reverse flow, air enters at the top (seg N-1) at ambient temperature
and flows downward. Each segment receives its upstream temperature from the
segment above (seg+1).

### A3. Seg 0 treatment during reverse flow

**Is it treated as an interior cell?** YES.

Lines 791–795:

```python
is_inlet_boundary = False
if mass_flow >= 0:
    is_inlet_boundary = (seg == 0)
else:
    is_inlet_boundary = (seg == chimney.total_segments - 1)
```

When `mdot < 0`, `is_inlet_boundary` is False for seg 0. Seg 0 enters the
main energy loop (line 797) and is processed identically to any other
interior segment.

**Is the bottom boundary temperature still used?** Only at initialization.

At line 121 of `solve_coupled_timestep`, before the Picard loop begins:

```python
current_state['temperatures'][0] = T_inlet
```

This sets seg 0 to `T_inlet` once. But inside each Picard iteration, the
energy solver computes a `new_temperatures[seg=0]` driven by
`T_upstream = temperatures[1]` (cold air from above). After relaxation
(line 159):

```python
T = alpha_picard * T_new + (1 - alpha_picard) * T_old
```

With `alpha_picard = 0.3` and 5 iterations, the initial `T_inlet` value at
seg 0 is progressively blended away. After 5 iterations, approximately
49–78% of the inlet-to-ambient temperature range is lost (see Section C).

**Is any heat source applied at the base?** NO.

There is no stove model, no combustion heat source, and no firebox thermal
mass in the simulation. The only heat input to the gas at seg 0 is:
- Wall convection/radiation (via h_inner), from the black pipe wall
- The initial T_inlet boundary condition, which is washed out by Picard

---

## Section B — Wall Heat Transfer Physics

### B4. Implemented heat-transfer mechanisms

All mechanisms computed in `comprehensive_heat_transfer.py`,
`calculate_segment_heat_transfer` method (lines 464–710):

| Mechanism | Location | Formula |
|-----------|----------|---------|
| **Forced convection** (gas-wall) | line 493 | Dittus-Boelter: `Nu = 0.023 Re^0.8 Pr^0.4` |
| **Natural convection** (gas-wall) | line 520 | `Nu = 0.56 Ra^0.25` (internal vertical pipe) |
| **Combined convection** | line 522 | Churchill-Usagi: `h = (h_f^3.5 + h_n^3.5)^(1/3.5)` |
| **Gas-wall radiation** (linearized) | line 547 | `h_rad = 4 * eps * sigma * T_avg^3`, added to h_inner |
| **Wall conduction** (radial, single) | line 370 | `R_wall = ln(r_o/r_i) / (2*pi*k*L)` |
| **Wall conduction** (triple-wall) | adaptive_triple_wall.py | 3-node network: liner-insulation-outer |
| **Outer convection** (wall-ambient) | line 587 | Nu correlations for external cylinder |
| **Outer radiation** (wall-ambient) | line 587–588 | `h_rad = eps*sigma*(T_w^2+T_a^2)(T_w+T_a)` |
| **Condensation** | line 556 | Film modification of h_inner + latent heat rate |

### B5. Does radiation contribute to gas heating or only wall losses?

**Radiation contributes directly to gas heating.** At line 547:

```python
h_inner += 4.0 * _eps_ir * self.STEFAN_BOLTZMANN * _T_avg_ir**3
```

The linearized radiation coefficient is added to `h_inner`, which then
appears in the gas energy equation. When `T_wall > T_gas`, this radiation
term drives heat FROM the wall TO the gas, warming it. It is not limited
to wall-to-ambient loss.

At typical 0227 backdraft conditions (T_wall ~ 370 K, T_gas ~ 295 K):
- h_rad ~ 6.7 W/(m2*K) — comparable to the natural convection coefficient

### B6. Wall temperature behavior at seg 0/seg 1 during backdraft

The wall energy equation for single-wall segments (lines 368–372 of
comprehensive_heat_transfer.py):

```
T_wall^{n+1} = [C_wall/dt * T_wall^n + UA_in * T_gas + UA_out * T_amb]
             / [C_wall/dt + UA_in + UA_out]
```

During backdraft:
- `UA_in = h_inner * A_inner` — drives wall toward gas temperature
- `UA_out = h_outer * A_outer` — drives wall toward ambient
- `C_wall/dt` — thermal inertia (resists rapid change)

The black pipe (24-gauge steel) has low thermal mass. Wall temperature
will track toward the gas temperature within a few minutes. If the gas is
cold (driven to ambient by advection), the wall follows — losing its ability
to warm the gas.

**Feedback loop:** cold gas cools the wall, which then cannot warm the gas.

---

## Section C — Advection vs Wall Heating Magnitude

### Representative conditions (0227 Phase 2, seg 0):

| Parameter | Value |
|-----------|-------|
| D_inner | 15.24 cm (6 in) |
| Segment length | 31.1 cm (12.25 in) |
| Wall surface area | 0.149 m2 |
| Gas mass in segment | 6.79 g |
| T_gas (during backdraft) | ~295 K (22 C) |
| T_wall (initially) | ~370 K (97 C) |
| T_upstream (from seg 1) | ~295 K (ambient) |
| mdot (backdraft) | -5 g/s |

### Heat transfer coefficients at seg 0:

| Component | Value (W/m2*K) |
|-----------|----------------|
| h_forced (Dittus-Boelter, Re=2300) | 1.7 |
| h_natural (Ra=2.8e7) | 6.9 |
| h_combined (Churchill-Usagi) | 7.0 |
| h_radiation (linearized) | 6.7 |
| **h_inner (total)** | **13.6** |

### Key energy rates:

```
Transport rate:  mdot * cp = 0.005 * 1006 = 5.03 W/K
Wall heating:    h * A     = 13.6 * 0.149 = 2.03 W/K

Ratio: transport / wall = 2.5x   --> advection dominates
```

When 5 g/s of cold air flows through this segment, it carries away heat
2.5 times faster than the wall can replenish it.

### T_steady from solver equation:

```
T_steady = T_upstream - (h*A*(T_gas - T_wall)) * (m_gas / mdot) / (m_gas * cp)
         = 295 - (-152.1) * 1.36 / (0.00679 * 1006)
         = 325.2 K = 52.1 C
```

Even in steady state with warm walls, the gas only reaches 52 C — well below
the stove inlet temperature of 277 C.

### Picard washout of T_inlet:

Seg 0 starts at T_inlet = 550 K (277 C) before the Picard loop. After 5
Picard iterations with alpha = 0.3:

| Iteration | T_seg0 (C) |
|-----------|------------|
| 0 (init) | 276.9 |
| 1 | 242.6 |
| 2 | 213.6 |
| 3 | 188.9 |
| 4 | 168.1 |
| 5 | 150.4 |

**49% of the inlet-ambient range is lost** within a single timestep.

### Sensitivity to backdraft rate:

| mdot (g/s) | transport/wall | T_steady (C) | After Picard (C) |
|------------|----------------|---------------|------------------|
| 1 | 0.50 | 97.9 | 208.2 |
| 3 | 1.49 | 72.3 | 175.7 |
| 5 | 2.48 | 52.1 | 150.4 |
| 10 | 4.96 | 37.0 | 119.8 |
| 20 | 9.92 | 29.4 | 96.7 |

At all typical backdraft rates, advection dominates. The wall can only
warm the gas 10–30 C above ambient.

---

## Section D — Stove Physics

### Current model representation of the stove:

| Feature | Present? | Notes |
|---------|----------|-------|
| Stove thermal mass | NO | No firebox heat capacity |
| Firebox wall temperature | NO | No temperature state variable |
| Stove-to-pipe conduction | NO | No heat path from firebox to chimney |
| Stove-to-gas radiation | NO | No radiative source below seg 0 |
| Residual combustion heat | NO | No heat source term |
| Stove surface temperature | NO | Not modeled |

**Confirmed:** The ONLY heat input to the gas occurs when `mdot > 0` via the
inlet boundary condition at seg 0 (line 121):

```python
current_state['temperatures'][0] = T_inlet
```

This is enforced once before Picard, and only effective during forward flow
because the inlet boundary treatment at seg 0 (line 834–835) preserves it.
During reverse flow, seg 0 is an interior cell and this initial value is
overwritten by the energy solver.

---

## Section E — Reverse-Flow Boundary Condition Logic

### Bottom boundary during reverse flow:

1. **T_inlet is set at seg 0 once** (line 121) before Picard loop starts.

2. **Picard loop runs 5+ iterations.** Each iteration:
   - Energy solver treats seg 0 as interior (not inlet boundary)
   - Computes `T_upstream = temperatures[1]` (cold air from above)
   - Computes `T_steady` between T_upstream and T_wall
   - Returns new temperature driven toward ambient
   - Picard relaxation blends: `T = 0.3 * T_new + 0.7 * T_old`

3. **Net effect:** T_inlet at seg 0 is washed out. After convergence, seg 0
   temperature reflects a balance between wall heating (weak) and advective
   cooling (strong).

### What the solver effectively assumes during reverse flow:

- Cold air enters at the top at ambient temperature: **YES** (line 747)
- No heat source at the bottom: **YES** (no stove model)
- No mixing region near the stove: **YES** (pure upwind scheme, no
  turbulent mixing or recirculation)

The model treats the bottom of the chimney as an open pipe end with no
thermal source. In reality, the stove firebox is a large (~50 kg) thermal
mass at 300–600 C radiating into the pipe entrance.

---

## Section F — Evidence from the 0227 Simulation

### Representative values during Phase 2

Based on analytical estimates using 0227 conditions (T_indoor = 21.4 C,
T_outdoor = 22.2 C, backdraft mdot ~ -5 g/s):

| Timestep | Seg | T_gas (C) | T_wall (C) | mdot (g/s) | T_upstream (C) | h_inner (W/m2K) | Q_wall (W) |
|----------|-----|-----------|------------|------------|----------------|-----------------|------------|
| X1 (~175s) | 0 | ~200 | ~300 | ~-2 | ~100 | ~13 | ~200 |
| X1 (~175s) | 1 | ~100 | ~250 | ~-2 | ~80 | ~12 | ~270 |
| mid-Phase2 (~300s) | 0 | ~30 | ~80 | ~-5 | ~25 | ~10 | ~75 |
| mid-Phase2 (~300s) | 1 | ~25 | ~60 | ~-5 | ~23 | ~8 | ~42 |
| pre-recovery (~400s) | 0 | ~23 | ~40 | ~-3 | ~22 | ~6 | ~15 |
| pre-recovery (~400s) | 1 | ~22 | ~35 | ~-3 | ~22 | ~6 | ~12 |

**Key observation:** By mid-Phase 2, both gas AND wall temperatures at the
bottom have collapsed to near-ambient. The wall's low thermal mass (24-gauge
steel = ~0.6 mm) means it cannot sustain heating for more than a few minutes
once the gas is cold.

The experimental S1 (15 inches above stove) reads ~50–80 C during this same
period — the real stove firebox keeps the bottom warm, but the model has no
mechanism to represent this.

---

## Section G — Final Diagnosis

### Why does S1 remain cold in the current model during backdraft?

**Answer: A combination of (A), (B), and (D).**

**(A) Missing stove heat source — PRIMARY CAUSE.**
The stove firebox has ~50 kg of thermal mass at 300–600 C. It radiates and
convects hundreds of watts into the chimney entrance. The model has zero
representation of this. During forward flow, this is implicitly captured by
the inlet boundary condition (T_inlet from TC1+offset or TC10). During
reverse flow, there is no equivalent mechanism.

**(B) Boundary condition treatment — SECONDARY CAUSE.**
Even though T_inlet is set at seg 0 before Picard, the energy solver treats
seg 0 as an interior cell during reverse flow and computes a new temperature
driven by cold upstream air. Picard relaxation washes out ~50% of T_inlet
per timestep. This is correct physics for a pipe with no heat source at the
bottom, but incorrect when a stove is present.

**(D) Advection overwhelming wall heating — CONTRIBUTING FACTOR.**
The transport rate (`mdot * cp = 5.03 W/K`) exceeds the wall heating rate
(`h * A = 2.03 W/K`) by 2.5x at typical backdraft rates. Even with warm
walls, the gas temperature can only rise 10–30 C above ambient. And the
thin steel walls cool within minutes, eliminating even this small effect.

**(C) Wall model limitations — MINOR CONTRIBUTOR.**
The wall model itself is adequate (single-wall lumped capacitance with
convection + radiation on both sides). The issue is not the wall model but
the wall's low thermal mass and the absence of an external heat source
(the stove) to keep the wall warm.

---

## Section H — Potential Physics Fixes (No Code Yet)

### Ranked by likely impact:

1. **Introduce stove thermal mass at the bottom boundary** (MOST LIKELY FIX)
   - Add a "firebox node" below seg 0 with configurable thermal mass,
     initial temperature, and heat transfer coefficient to seg 0.
   - During reverse flow, this node radiates/convects heat into the gas
     at seg 0, preventing collapse to ambient.
   - During forward flow, this node is bypassed (gas enters at T_inlet).
   - Parameters: firebox mass (~50 kg), surface area (~0.3 m2),
     emissivity (~0.9), initial temperature (from TC10 or T_inlet).

2. **Modify bottom boundary condition during reverse flow**
   - Instead of treating seg 0 as an interior cell, impose a mixed
     boundary condition: `T_seg0 = max(T_energy_solver, T_stove_surface)`
   - Simpler than a full firebox model but less physically rigorous.

3. **Introduce a mixing/recirculation region near the base**
   - During reverse flow, flow near the stove is not pure downward pipe
     flow. The firebox creates a recirculation zone where hot and cold
     air mix. A mixing-length model could partially represent this.

4. **Modify wall-gas coupling**
   - Increase effective wall thermal mass at seg 0 to account for the
     thermal coupling between the stove and the pipe.
   - Less physical but computationally simple.

### NOT required:

- Changes to the radiation model (already included in h_inner)
- Changes to the convection correlations (correct for pipe flow)
- Changes to the Picard iteration scheme (correct numerical method;
  the issue is physics, not numerics)
- Changes to the upwind advection scheme (correct for 1D pipe flow)
