# Merge Log: Safe Physics Merge — 2026-04-03

## Source Branch
- **Branch**: `fix/external-convection-length-scale`
- **Branch HEAD (committed)**: `d3a36dfb90b9e7c01bd6d65827c12be6f4f530eb`
- **Branch HEAD (with uncommitted session changes)**: will be committed as a new commit before merge

## Target Branch
- **Branch**: `master`
- **Master HEAD**: `50569c35b7e9bd098cef7246195d2ddcdfbdc4f6`

## Backup References
- Tag `pre-merge/branch-head`: branch HEAD before merge
- Tag `pre-merge/master-head`: master HEAD before merge

## Approved Changes to Merge

### 1. External convection length scale + film temperature
- **Files**: `flued_module/comprehensive_heat_transfer.py`
- Vertical natural convection uses `part_height` (not diameter) as Gr length scale
- Air properties at film temperature `(T_wall + T_ambient)/2`
- Applied to both single-wall and triple-wall sections
- Standard per Incropera / Churchill & Chu

### 2. Internal convection turbulent entry-region correction
- **Files**: `flued_module/comprehensive_heat_transfer.py`
- Gnielinski (1976): `Nu = Nu_fd × [1 + (D/x)^(2/3)]` for L/D < 60
- x = cumulative distance from part entrance (resets at fittings)
- Safeguard: only applied when `length_diameter_ratio` is provided and < 60
- Existing fitting augmentation (×1.2) is separate, not double-counted

### 3. Coupled solver inlet pressure boundary fix
- **Files**: `flued_module/coupled_solver.py`
- `new_pressures[0] = Pa - dp_inlet_drop` (was `Pa`)
- Physically correct: gas enters below atmospheric due to inlet restriction
- Correctly changes leakage sign convention (infiltration at bottom)

### 4. Pressure reconstruction — dp_door not subtracted
- **Files**: `flued_module/chimney.py`
- `get_static_pressure_profile` no longer subtracts dp_door/dp_stove uniformly
- Was double-counting: momentum solver already accounts for dp_door via reduced mass flow
- Confirmed experimentally: 0130 test TC3/TOP match perfectly without subtraction
- dp_door/dp_stove still computed and returned as diagnostics
- dp_room IS still subtracted (needed for depressurized room boundary condition)

### 5. Per-segment ambient pressure column
- **Files**: `flued_module/chimney.py`
- P_ambient computed segment-by-segment using each part's T_ambient_K and Pa
- Replaces single average rho_ambient across all parts
- More accurate for mixed indoor/outdoor chimney sections

### 6. Draft analysis fix
- **Files**: `flued_module/draft_analysis.py`
- Buoyancy uses `segment_vertical_heights` (vertical rise only)
- Friction uses `segment_lengths` (flow path length)
- Previously both used `height_m / num_segments` (wrong for horizontal/elbow sections)

### 7. Supporting infrastructure (needed by above)
- `x_from_part_start` in segment_data (`chimney.py`, `coupled_solver.py`)
- `part_height` in segment_data (`chimney.py`, `coupled_solver.py`)
- `K_entrance` field in InletRestrictionConfig (defined, not used — placeholder)
- `K_exit` support in top boundary and pressure reconstruction
- `effective_area_uncapped_m2` diagnostic field
- Callable `room_pressure_offset_Pa` for time-varying depressurization
- Pressure decomposition return values from `get_static_pressure_profile`

## Changes Explicitly EXCLUDED from This Merge

### A. Rain cap Cp change
- **File**: `flued_module/wind_cap_effects.py`
- Changed `rain_cap` Cp from `+0.05` to `-0.30`
- **Reason**: Needs further justification. -0.30 is aggressive. Self-test assertion contradicts the sign.

### B. Tee K-factor reduction
- **File**: `flued_module/flow/elbows.py`
- Tee K reduced from 0.9-1.1 to 0.5-0.7
- K_override feature added to base_K()
- **Reason**: ~40% reduction needs specific ASHRAE reference. Both ranges are within published bounds.

### C. mass_flow.py comment changes
- **File**: `flued_module/mass_flow.py`
- Comment-only changes (removed "VENT-II" references)
- **Reason**: Cosmetic, not needed for physics merge.

### D. Calibration framework
- **Files**: `flued_module/calibration/` (new directory)
- **Reason**: Separate feature, not physics fixes.
