"""Staged calibration runner with caching and JSON serialization."""

import json
import hashlib
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple
from scipy.optimize import minimize

from .param_space import (
    StageAParams, StageBParams, StageCParams, StageDParams,
    latin_hypercube_sample,
)


def _param_hash(params_dict: dict) -> str:
    """Deterministic hash of a parameter dict for caching."""
    s = json.dumps(params_dict, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


class StagedCalibrator:
    """Run parameter sweeps and local refinement for a given stage.

    Parameters
    ----------
    out_dir : Path
        Output directory for this stage (e.g., outputs/calibration/<ts>/stage_A/).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, out_dir: Path, seed: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self.cache: Dict[str, dict] = {}   # hash -> result dict
        self._load_cache()

    def _cache_path(self) -> Path:
        return self.out_dir / 'cache.json'

    def _load_cache(self):
        p = self._cache_path()
        if p.exists():
            with open(p) as f:
                self.cache = json.load(f)

    def _save_cache(self):
        with open(self._cache_path(), 'w') as f:
            json.dump(self.cache, f, indent=2, default=str)

    def coarse_sweep(
        self,
        param_cls,
        n_samples: int,
        eval_fn: Callable,
        verbose: bool = True,
    ) -> List[dict]:
        """Latin hypercube sweep over parameter space.

        Parameters
        ----------
        param_cls : StageAParams, StageBParams, etc.
            Dataclass with .from_array(), .bounds_arrays(), .names(), .validate()
        n_samples : int
            Number of LHS samples.
        eval_fn : callable
            eval_fn(params_dict) -> (objective_value, details_dict)
            Must accept a dict of param name -> value.
        verbose : bool

        Returns
        -------
        results : list of dicts, each with 'params', 'objective', 'details', 'valid'
        """
        lo, hi = param_cls.bounds_arrays()
        samples = latin_hypercube_sample(n_samples, lo, hi, self.rng)
        results = []

        for i, x in enumerate(samples):
            params = param_cls.from_array(x)
            pd = params.to_dict()
            h = _param_hash(pd)

            if h in self.cache:
                r = self.cache[h]
                if verbose:
                    print(f'  [{i+1}/{n_samples}] CACHED  J={r["objective"]:.4f}  {pd}')
                results.append(r)
                continue

            if not params.validate():
                r = {'params': pd, 'objective': 1e12, 'details': {}, 'valid': False, 'hash': h}
                if verbose:
                    print(f'  [{i+1}/{n_samples}] INVALID {pd}')
                results.append(r)
                continue

            t0 = time.time()
            try:
                obj, details = eval_fn(pd)
                elapsed = time.time() - t0
                r = {
                    'params': pd, 'objective': float(obj),
                    'details': details, 'valid': True, 'hash': h,
                    'elapsed_s': round(elapsed, 1),
                }
            except Exception as e:
                elapsed = time.time() - t0
                r = {
                    'params': pd, 'objective': 1e12,
                    'details': {'error': str(e)}, 'valid': False, 'hash': h,
                    'elapsed_s': round(elapsed, 1),
                }

            self.cache[h] = r
            results.append(r)
            if verbose:
                tag = 'OK' if r['valid'] else 'FAIL'
                print(f'  [{i+1}/{n_samples}] {tag}  J={r["objective"]:.4f}  '
                      f'({elapsed:.0f}s)  {pd}')

        self._save_cache()
        return results

    def local_refine(
        self,
        param_cls,
        x0: np.ndarray,
        eval_fn: Callable,
        method: str = 'Nelder-Mead',
        maxiter: int = 40,
        verbose: bool = True,
    ) -> dict:
        """Local optimization starting from x0.

        Parameters
        ----------
        param_cls : stage parameter class
        x0 : initial guess array
        eval_fn : callable(params_dict) -> (objective, details)
        method : scipy.optimize method
        maxiter : max iterations

        Returns
        -------
        result dict with 'params', 'objective', 'details', 'scipy_result'
        """
        lo, hi = param_cls.bounds_arrays()
        names = param_cls.names()
        best = {'objective': 1e12}

        def _obj(x):
            # Clip to bounds
            x_c = np.clip(x, lo, hi)
            params = param_cls.from_array(x_c)
            pd = params.to_dict()
            h = _param_hash(pd)

            if h in self.cache and self.cache[h]['valid']:
                obj = self.cache[h]['objective']
            else:
                if not params.validate():
                    return 1e12
                try:
                    obj, details = eval_fn(pd)
                    self.cache[h] = {
                        'params': pd, 'objective': float(obj),
                        'details': details, 'valid': True, 'hash': h,
                    }
                except Exception:
                    return 1e12

            if obj < best.get('objective', 1e12):
                best['objective'] = obj
                best['params'] = pd
                if verbose:
                    vals = '  '.join(f'{n}={pd[n]:.4f}' for n in names)
                    print(f'    refine: J={obj:.4f}  {vals}')
            return obj

        bounds = list(zip(lo, hi))
        res = minimize(_obj, x0, method=method,
                       options={'maxiter': maxiter, 'xatol': 1e-3, 'fatol': 1e-4,
                                'adaptive': True})
        self._save_cache()

        x_best = np.clip(res.x, lo, hi)
        params_best = param_cls.from_array(x_best)
        pd_best = params_best.to_dict()
        h = _param_hash(pd_best)
        cached = self.cache.get(h, {})

        return {
            'params': pd_best,
            'objective': float(res.fun),
            'details': cached.get('details', {}),
            'scipy_result': {
                'success': bool(res.success),
                'message': res.message,
                'nfev': int(res.nfev),
                'nit': int(res.nit) if hasattr(res, 'nit') else None,
            },
        }

    def get_top_n(self, results: List[dict], n: int = 10) -> List[dict]:
        """Return top N results sorted by objective (ascending)."""
        valid = [r for r in results if r.get('valid', False)]
        return sorted(valid, key=lambda r: r['objective'])[:n]

    def save_results(self, results: List[dict], filename: str = 'results.json'):
        """Save results list to JSON."""
        p = self.out_dir / filename
        # Convert numpy types for JSON serialization
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(p, 'w') as f:
            json.dump(results, f, indent=2, default=_convert)
        return p

    def save_best_params(self, params_dict: dict, filename: str = 'best_params.json'):
        """Save best parameter set to JSON."""
        p = self.out_dir / filename
        with open(p, 'w') as f:
            json.dump(params_dict, f, indent=2)
        return p
