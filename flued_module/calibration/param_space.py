"""Parameter sets and bounds for staged calibration."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional
import numpy as np


# ------------------------------------------------------------------ #
#  Stage A: Momentum / resistance (closed-door)
# ------------------------------------------------------------------ #
@dataclass
class StageAParams:
    A_leak_cm2: float = 75.0       # inlet leak area (cm^2)
    Cd: float = 0.62               # inlet discharge coefficient
    K_stove: float = 0.0           # stove body minor-loss coefficient
    K_minor_mult: float = 1.0      # multiplier on all fitting K values (via vane_factor)

    BOUNDS: dict = field(default_factory=lambda: {
        'A_leak_cm2':  (10.0, 150.0),
        'Cd':          (0.3, 1.0),
        'K_stove':     (0.0, 5.0),
        'K_minor_mult': (0.2, 5.0),
    }, repr=False, compare=False)

    def validate(self) -> bool:
        for name, (lo, hi) in self.BOUNDS.items():
            v = getattr(self, name)
            if v < lo or v > hi:
                return False
        return True

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != 'BOUNDS'}

    def to_array(self) -> np.ndarray:
        return np.array([self.A_leak_cm2, self.Cd, self.K_stove, self.K_minor_mult])

    @classmethod
    def from_array(cls, x: np.ndarray) -> 'StageAParams':
        return cls(A_leak_cm2=float(x[0]), Cd=float(x[1]),
                   K_stove=float(x[2]), K_minor_mult=float(x[3]))

    @classmethod
    def names(cls) -> list:
        return ['A_leak_cm2', 'Cd', 'K_stove', 'K_minor_mult']

    @classmethod
    def bounds_arrays(cls) -> Tuple[np.ndarray, np.ndarray]:
        b = cls().BOUNDS
        lo = np.array([b[n][0] for n in cls.names()])
        hi = np.array([b[n][1] for n in cls.names()])
        return lo, hi


# ------------------------------------------------------------------ #
#  Stage B: Heat-loss (closed-door)
# ------------------------------------------------------------------ #
@dataclass
class StageBParams:
    h_inner_mult: float = 1.0     # multiplier on internal convection h
    h_outer_mult: float = 1.0     # multiplier on external convection h
    eps_inner_scale: float = 1.0  # multiplier on inner-wall emissivity

    BOUNDS: dict = field(default_factory=lambda: {
        'h_inner_mult':   (0.5, 3.0),
        'h_outer_mult':   (0.5, 3.0),
        'eps_inner_scale': (0.5, 2.0),
    }, repr=False, compare=False)

    def validate(self) -> bool:
        for name, (lo, hi) in self.BOUNDS.items():
            v = getattr(self, name)
            if v < lo or v > hi:
                return False
        return True

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != 'BOUNDS'}

    def to_array(self) -> np.ndarray:
        return np.array([self.h_inner_mult, self.h_outer_mult, self.eps_inner_scale])

    @classmethod
    def from_array(cls, x: np.ndarray) -> 'StageBParams':
        return cls(h_inner_mult=float(x[0]), h_outer_mult=float(x[1]),
                   eps_inner_scale=float(x[2]))

    @classmethod
    def names(cls) -> list:
        return ['h_inner_mult', 'h_outer_mult', 'eps_inner_scale']

    @classmethod
    def bounds_arrays(cls) -> Tuple[np.ndarray, np.ndarray]:
        b = cls().BOUNDS
        lo = np.array([b[n][0] for n in cls.names()])
        hi = np.array([b[n][1] for n in cls.names()])
        return lo, hi


# ------------------------------------------------------------------ #
#  Stage C: Cap / wind sensitivity
# ------------------------------------------------------------------ #
@dataclass
class StageCParams:
    Cp_override: float = -0.30    # wind-cap pressure coefficient

    BOUNDS: dict = field(default_factory=lambda: {
        'Cp_override': (-0.80, 0.10),
    }, repr=False, compare=False)

    def validate(self) -> bool:
        for name, (lo, hi) in self.BOUNDS.items():
            v = getattr(self, name)
            if v < lo or v > hi:
                return False
        return True

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != 'BOUNDS'}

    def to_array(self) -> np.ndarray:
        return np.array([self.Cp_override])

    @classmethod
    def from_array(cls, x: np.ndarray) -> 'StageCParams':
        return cls(Cp_override=float(x[0]))

    @classmethod
    def names(cls) -> list:
        return ['Cp_override']

    @classmethod
    def bounds_arrays(cls) -> Tuple[np.ndarray, np.ndarray]:
        b = cls().BOUNDS
        lo = np.array([b[n][0] for n in cls.names()])
        hi = np.array([b[n][1] for n in cls.names()])
        return lo, hi


# ------------------------------------------------------------------ #
#  Stage D: Cracked-door refinement
# ------------------------------------------------------------------ #
@dataclass
class StageDParams:
    door_area_scale: float = 1.0   # multiplier on cracked-door effective area
    door_ramp_s: float = 5.0       # ramp time for door transitions

    BOUNDS: dict = field(default_factory=lambda: {
        'door_area_scale': (0.3, 3.0),
        'door_ramp_s':     (0.5, 10.0),
    }, repr=False, compare=False)

    def validate(self) -> bool:
        for name, (lo, hi) in self.BOUNDS.items():
            v = getattr(self, name)
            if v < lo or v > hi:
                return False
        return True

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != 'BOUNDS'}

    def to_array(self) -> np.ndarray:
        return np.array([self.door_area_scale, self.door_ramp_s])

    @classmethod
    def from_array(cls, x: np.ndarray) -> 'StageDParams':
        return cls(door_area_scale=float(x[0]), door_ramp_s=float(x[1]))

    @classmethod
    def names(cls) -> list:
        return ['door_area_scale', 'door_ramp_s']

    @classmethod
    def bounds_arrays(cls) -> Tuple[np.ndarray, np.ndarray]:
        b = cls().BOUNDS
        lo = np.array([b[n][0] for n in cls.names()])
        hi = np.array([b[n][1] for n in cls.names()])
        return lo, hi


# ------------------------------------------------------------------ #
#  Latin hypercube sampling
# ------------------------------------------------------------------ #
def latin_hypercube_sample(n_samples: int, lo: np.ndarray, hi: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Generate n_samples in [lo, hi] using Latin hypercube sampling.

    Returns array of shape (n_samples, n_dims).
    """
    n_dims = len(lo)
    result = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            lo_bin = perm[i] / n_samples
            hi_bin = (perm[i] + 1) / n_samples
            u = rng.uniform(lo_bin, hi_bin)
            result[i, d] = lo[d] + u * (hi[d] - lo[d])
    return result
