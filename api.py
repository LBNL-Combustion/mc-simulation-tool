"""
FastAPI server for flue simulation.

Start:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              - health check
    POST /simulate            - run simulation (blocking, ~60s)
    POST /simulate/async      - submit simulation job, returns job_id
    GET  /simulate/status/{id} - poll job status / get results
    GET  /sample-config       - return sample_config.json for reference
"""
import uuid
import json
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from run_simulation import run_simulation

app = FastAPI(title="Flue Simulation API", version="1.0.0")

# Allow cross-origin requests from any front-end dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Async job infrastructure
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=2)
_jobs: dict = {}        # job_id -> {status, result, error}
_jobs_lock = Lock()


def _run_job(job_id: str, config: dict):
    """Run simulation in background thread and store result."""
    try:
        result = run_simulation(config)
        with _jobs_lock:
            _jobs[job_id]['status'] = 'completed'
            _jobs[job_id]['result'] = result
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'failed'
            _jobs[job_id]['error'] = f"{type(e).__name__}: {e}"
            _jobs[job_id]['traceback'] = traceback.format_exc()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/simulate")
def simulate_sync(config: dict):
    """Run simulation and return results. Blocks for ~60s."""
    try:
        results = run_simulation(config)
        return results
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing required field: {e}")
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/simulate/async")
def simulate_async(config: dict):
    """Submit a simulation job. Returns immediately with a job_id."""
    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {'status': 'running', 'result': None, 'error': None}
    _executor.submit(_run_job, job_id, config)
    return {"job_id": job_id, "status": "running"}


@app.get("/simulate/status/{job_id}")
def simulate_status(job_id: str):
    """Check status of an async simulation job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job['status'] == 'running':
        return {"job_id": job_id, "status": "running"}
    elif job['status'] == 'completed':
        return {"job_id": job_id, "status": "completed", "result": job['result']}
    else:
        return {"job_id": job_id, "status": "failed", "error": job['error']}


@app.get("/sample-config")
def sample_config():
    """Return the sample config JSON for reference."""
    sample_path = Path(__file__).parent / "sample_config.json"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="sample_config.json not found")
    with open(sample_path) as f:
        return json.load(f)
