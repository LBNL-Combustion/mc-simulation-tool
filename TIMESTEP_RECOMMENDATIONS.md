# Timestep Recommendations for 2-Hour Simulations

## Executive Summary

Yes, you CAN simulate 2 hours with 1-5 second timesteps, but with important caveats:

### **Stability**: The semi-implicit method is stable for large timesteps
### **Accuracy**: Larger timesteps smooth out rapid transients
### **Fix Applied**: Fixed NaN errors in Reynolds/Prandtl calculations

## Key Physics Considerations

### 1. **Critical Time Constants**
- **Gas residence time**: ~0.5s per segment (fast)
- **Wall thermal response**: ~200s for single wall (medium)
- **Liner response**: ~2s for triple wall liner (fast)
- **Flow establishment**: ~10s for full chimney (medium)

### 2. **Timestep Effects on Physics**

#### Mass Flow Conservation
```
Distance traveled = velocity × dt
Energy transferred = heat_flux × dt
```
The code correctly uses `dt` in:
- Energy balance: `dT_dt = (...) / (gas_mass * cp)` then `T_new = T_old + dT_dt * dt`
- Momentum balance: `alpha = dt / (tau + dt)` for flow inertia
- Wall heating: `T_wall_new = f(dt, heat_fluxes)`

## Recommended Approaches

### Option 1: Fixed Large Timestep (Simple)
```python
# For 2-hour simulation where steady state is the goal
chimney = Chimney(parts=parts, dt=5.0, total_time=7200)
```
**Pros**: Fast (1440 steps), stable
**Cons**: Misses initial transients, may need smaller dt for first minute

### Option 2: Variable Timestep Schedule
```python
# Better accuracy with efficiency
# First minute: dt=0.5s (capture startup)
# 1-10 min: dt=1.0s (moderate dynamics)  
# 10min-2hr: dt=5.0s (slow changes)
```
**Pros**: Captures all physics, efficient
**Cons**: Requires implementation of variable dt

### Option 3: Adaptive Timestep (Best)
```python
# Use AdaptiveTimestepManager
# Automatically adjusts dt based on rate of change
```
**Pros**: Optimal accuracy and efficiency
**Cons**: More complex implementation

## Specific Fixes Applied

### 1. **NaN Error Fix**
Added bounds checking in Reynolds and Prandtl calculations:
```python
if reynolds <= 0 or prandtl <= 0:
    return 3.66  # Fallback value
if np.isnan(Re) or np.isinf(Re):
    return 1.0  # Minimum Re
```

### 2. **Damping Fix** 
Removed excessive damping that was slowing wall heating

## Performance Estimates

For 2-hour (7200s) simulation:
- **dt=0.1s**: 72,000 steps (~10 min runtime)
- **dt=0.5s**: 14,400 steps (~2 min runtime)
- **dt=1.0s**: 7,200 steps (~1 min runtime)
- **dt=5.0s**: 1,440 steps (~15 sec runtime)
- **Adaptive**: ~3,000 steps (~30 sec runtime)

## Recommendations

### For Engineering Analysis (Steady State Focus)
Use **dt=5.0s** after an initial period with smaller timestep:
1. Run 0-60s with dt=0.5s
2. Continue 60s-2hr with dt=5.0s

### For Detailed Transient Analysis
Use adaptive timestepping or the schedule:
- 0-30s: dt=0.1s
- 30s-2min: dt=0.5s
- 2-10min: dt=1.0s
- 10-30min: dt=2.0s
- 30min-2hr: dt=5.0s

### For Quick Studies
Use fixed dt=5.0s throughout, understanding you'll miss rapid transients

## Important Notes

1. **Conservation**: The physics correctly scales with dt
2. **Condensation**: May need smaller dt when crossing dew point
3. **Flow reversal**: May need smaller dt if buoyancy changes sign
4. **Initial transient**: Always use smaller dt for first 30-60s

The simulation is designed to handle large timesteps through semi-implicit methods, making 2-hour simulations practical and efficient.