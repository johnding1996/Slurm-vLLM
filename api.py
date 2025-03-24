#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from datetime import datetime
import statistics
import random
import logging
from enum import Enum, auto

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from tabulate import tabulate
import uvicorn
import httpx

# Configure logging to suppress access logs
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)


# Backend health states
class BackendState(Enum):
    HEALTHY = auto()  # Fully operational
    DEGRADED = auto()  # Experienced failures but may recover
    UNHEALTHY = auto()  # Multiple failures, under recovery monitoring
    INITIALIZING = auto()  # New backend, not yet proven healthy
    DEAD = auto()  # Permanently removed (not in Slurm output)


# ANSI color codes for colorful output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def colorize(text, color):
    """Add color to text."""
    return f"{color}{text}{Colors.ENDC}"


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create background task for monitoring
    monitoring_task = asyncio.create_task(monitor_backends())
    print(colorize("API Proxy started! Monitoring backends...", Colors.GREEN))

    yield  # This is where FastAPI runs

    # Shutdown: cancel monitoring task
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        print(colorize("Backend monitoring stopped.", Colors.YELLOW))


# FastAPI app with lifespan
app = FastAPI(title="vLLM API Proxy", lifespan=lifespan)

# Global variables for tracking backends
backends = (
    {}
)  # model_index -> list of {"node": str, "port": int, "model_name": str, "job_name": str, "latency": float}
model_names = {}  # model_index -> model_name
recent_latencies = {}  # node:port -> list of recent latencies
backend_failures = {}  # node:port -> count of consecutive failures
backend_successes = {}  # node:port -> count of consecutive successful health checks
backend_requests = {}  # node:port -> count of requests sent
backend_states = {}  # node:port -> BackendState
backend_in_flight = {}  # node:port -> count of in-flight requests
backend_check_times = {}  # node:port -> next check time
backend_discovery_times = {}  # node:port -> when backend was first discovered

# Monitoring settings
MONITOR_INTERVAL = 5  # seconds
MAX_LATENCY_SAMPLES = 10
FAILURE_THRESHOLD = 10  # consecutive failures before marking as UNHEALTHY
SUCCESS_THRESHOLD = 3  # consecutive successes before promoting from DEGRADED to HEALTHY
BACKEND_TIMEOUT = 10  # seconds for health check timeout
INITIAL_GRACE_PERIOD = 0  # No grace period - backends should be checked immediately
MAX_INITIALIZING_TIME = 180  # seconds before INITIALIZING -> UNHEALTHY
REQUEST_STREAM_TIMEOUT = 60  # seconds to wait for first token in streaming response
MAX_RETRIES = 3  # maximum number of retries for a failed request
DEGRADED_CHECK_INTERVAL = 3  # seconds between DEGRADED backend checks
UNHEALTHY_CHECK_INTERVAL = 10  # initial seconds between UNHEALTHY backend checks
INITIALIZING_CHECK_INTERVAL = (
    0  # seconds between INITIALIZING backend checks - check immediately
)
MAX_BACKOFF_INTERVAL = 60  # maximum seconds for exponential backoff

# Model regex pattern
MODEL_PATTERN = (
    r"^v(\d{2})(\d{2})(\d{3})$"  # Format: v[model_index][job_id][port_suffix]
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=colorize("vLLM API Proxy with Smart Load Balancing", Colors.HEADER)
    )
    parser.add_argument(
        "--port", type=int, default=9090, help="Port to run the API proxy server on"
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=5,
        help="Interval in seconds between backend job checks",
    )

    return parser.parse_args()


def extract_job_info(job_name):
    """Extract model index, job ID and port from job name."""
    match = re.match(MODEL_PATTERN, job_name)
    if match:
        model_index = int(match.group(1))
        job_id = int(match.group(2))
        port_suffix = match.group(3)
        port = int("9" + port_suffix)  # Add "9" prefix to make it 9XXX
        return model_index, job_id, port
    return None, None, None


async def check_backend_health(node, port):
    """Check if a backend is healthy by making a simple request."""
    url = f"http://{node}:{port}/v1/models"
    backend_key = f"{node}:{port}"

    try:
        async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
            start_time = time.time()
            response = await client.get(url)
            latency = time.time() - start_time

            if response.status_code == 200:
                # Update latency information
                if backend_key not in recent_latencies:
                    recent_latencies[backend_key] = []

                recent_latencies[backend_key].append(latency)
                # Keep only the most recent samples
                if len(recent_latencies[backend_key]) > MAX_LATENCY_SAMPLES:
                    recent_latencies[backend_key] = recent_latencies[backend_key][
                        -MAX_LATENCY_SAMPLES:
                    ]

                # Update health tracking
                backend_failures[backend_key] = 0
                backend_successes[backend_key] = (
                    backend_successes.get(backend_key, 0) + 1
                )

                # State transitions based on consecutive successes
                current_state = backend_states.get(
                    backend_key, BackendState.INITIALIZING
                )

                if (
                    current_state == BackendState.DEGRADED
                    and backend_successes[backend_key] >= SUCCESS_THRESHOLD
                ):
                    # Promote from DEGRADED to HEALTHY after consecutive successes
                    backend_states[backend_key] = BackendState.HEALTHY
                    print(
                        colorize(
                            f"Backend {backend_key} recovered: DEGRADED → HEALTHY",
                            Colors.GREEN,
                        )
                    )

                elif (
                    current_state == BackendState.UNHEALTHY
                    and backend_successes[backend_key] >= SUCCESS_THRESHOLD
                ):
                    # Promote from UNHEALTHY to DEGRADED after consecutive successes
                    backend_states[backend_key] = BackendState.DEGRADED
                    print(
                        colorize(
                            f"Backend {backend_key} improving: UNHEALTHY → DEGRADED",
                            Colors.YELLOW,
                        )
                    )

                elif current_state == BackendState.INITIALIZING:
                    # New backend is now healthy
                    backend_states[backend_key] = BackendState.HEALTHY
                    print(
                        colorize(
                            f"Backend {backend_key} ready: INITIALIZING → HEALTHY",
                            Colors.GREEN,
                        )
                    )

                # Update check time for next health check based on state
                update_next_check_time(backend_key)

                return True, latency
            else:
                # Handle failure
                return handle_health_check_failure(
                    backend_key, f"HTTP {response.status_code}"
                )
    except Exception as e:
        # Handle exception
        return handle_health_check_failure(backend_key, str(e))


def update_next_check_time(backend_key):
    """Update when to next check a backend based on its current state."""
    current_state = backend_states.get(backend_key, BackendState.HEALTHY)

    if current_state == BackendState.DEGRADED:
        # Check DEGRADED backends frequently
        backend_check_times[backend_key] = time.time() + DEGRADED_CHECK_INTERVAL

    elif current_state == BackendState.UNHEALTHY:
        # Use exponential backoff for UNHEALTHY backends, with a maximum interval
        failures = backend_failures.get(backend_key, 0)
        backoff = min(
            UNHEALTHY_CHECK_INTERVAL * (2 ** (failures - FAILURE_THRESHOLD)),
            MAX_BACKOFF_INTERVAL,
        )
        backend_check_times[backend_key] = time.time() + backoff

    elif current_state == BackendState.INITIALIZING:
        # Check INITIALIZING backends frequently
        backend_check_times[backend_key] = time.time() + INITIALIZING_CHECK_INTERVAL


def handle_health_check_failure(backend_key, error_msg):
    """Handle a failed health check and update backend state accordingly."""
    # Increment failure count and reset success count
    backend_failures[backend_key] = backend_failures.get(backend_key, 0) + 1
    backend_successes[backend_key] = 0

    current_state = backend_states.get(backend_key, BackendState.HEALTHY)

    # State transitions based on failures
    if current_state == BackendState.HEALTHY:
        if backend_failures[backend_key] == 1:
            # First failure for a healthy backend -> DEGRADED
            backend_states[backend_key] = BackendState.DEGRADED
            print(
                colorize(
                    f"Backend {backend_key} degraded: HEALTHY → DEGRADED ({error_msg})",
                    Colors.YELLOW,
                )
            )

    elif current_state == BackendState.DEGRADED:
        if backend_failures[backend_key] >= FAILURE_THRESHOLD:
            # Too many failures for a degraded backend -> UNHEALTHY
            backend_states[backend_key] = BackendState.UNHEALTHY
            print(
                colorize(
                    f"Backend {backend_key} unhealthy: DEGRADED → UNHEALTHY ({error_msg})",
                    Colors.RED,
                )
            )

    elif current_state == BackendState.INITIALIZING:
        # Check if we've been trying to initialize for too long
        discovery_time = backend_discovery_times.get(backend_key, time.time())
        initializing_time = time.time() - discovery_time

        if initializing_time > INITIAL_GRACE_PERIOD + MAX_INITIALIZING_TIME:
            # Backend failed to initialize within time limit
            backend_states[backend_key] = BackendState.UNHEALTHY
            print(
                colorize(
                    f"Backend {backend_key} failed to initialize after {int(initializing_time)}s: INITIALIZING → UNHEALTHY",
                    Colors.RED,
                )
            )

    # Update check time for next health check
    update_next_check_time(backend_key)

    return False, None


def get_avg_latency(node, port):
    """Get the average latency for a backend."""
    if f"{node}:{port}" in recent_latencies and recent_latencies[f"{node}:{port}"]:
        return statistics.mean(recent_latencies[f"{node}:{port}"])
    return float("inf")  # Return infinity if no latency data is available


async def get_vllm_jobs():
    """Get all vLLM jobs using Slurm."""
    username = os.environ.get("USER", subprocess.getoutput("whoami"))

    # Get job information with node assignments
    cmd = ["squeue", "-u", username, "-h", "-o", "%j %N"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    jobs = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 2:
            job_name = parts[0]
            node = parts[1]

            if job_name.startswith("v"):
                model_index, job_id, port = extract_job_info(job_name)
                if model_index is not None:
                    jobs.append(
                        {
                            "job_name": job_name,
                            "node": node,
                            "model_index": model_index,
                            "job_id": job_id,
                            "port": port,
                        }
                    )

    return jobs


async def update_backend_info():
    """Update backend information by checking Slurm jobs and managing backend lifecycle."""
    global backends, model_names

    # Get current vLLM jobs from Slurm
    jobs = await get_vllm_jobs()

    # Create a set of all current backend keys from Slurm
    current_backend_keys = set()
    for job in jobs:
        current_backend_keys.add(f"{job['node']}:{job['port']}")

    # Identify missing backends (removed from Slurm output)
    for model_index in list(backends.keys()):
        for backend in list(backends[model_index]):
            backend_key = f"{backend['node']}:{backend['port']}"

            if backend_key not in current_backend_keys:
                # Backend no longer in Slurm output - mark as DEAD and remove
                print(
                    colorize(
                        f"Backend {backend_key} no longer in Slurm output - marked as DEAD",
                        Colors.RED,
                    )
                )
                backends[model_index].remove(backend)
                backend_states[backend_key] = BackendState.DEAD

                # Clean up associated resources
                if backend_key in recent_latencies:
                    del recent_latencies[backend_key]
                if backend_key in backend_failures:
                    del backend_failures[backend_key]
                if backend_key in backend_successes:
                    del backend_successes[backend_key]
                if backend_key in backend_check_times:
                    del backend_check_times[backend_key]
                if backend_key in backend_discovery_times:
                    del backend_discovery_times[backend_key]

        # Remove empty model entries
        if not backends[model_index]:
            del backends[model_index]
            if model_index in model_names:
                del model_names[model_index]

    # Process each job from Slurm
    for job in jobs:
        model_index = job["model_index"]
        node = job["node"]
        port = job["port"]
        backend_key = f"{node}:{port}"

        # Initialize if this model_index is new
        if model_index not in backends:
            backends[model_index] = []

        # Check if this backend is already in our list
        existing = False
        for backend in backends[model_index]:
            if backend["node"] == node and backend["port"] == port:
                existing = True
                break

        if not existing:
            # This is a new backend from Slurm
            print(
                colorize(
                    f"Discovered new backend {backend_key} for model index {model_index}",
                    Colors.CYAN,
                )
            )

            # Record discovery time
            backend_discovery_times[backend_key] = time.time()

            # Mark as INITIALIZING - but will be checked immediately
            backend_states[backend_key] = BackendState.INITIALIZING
            backend_failures[backend_key] = 0
            backend_successes[backend_key] = 0

            # Set check time to now - no grace period
            backend_check_times[backend_key] = time.time()

            # Add to backends with placeholder latency
            backends[model_index].append(
                {
                    "node": node,
                    "port": port,
                    "job_name": job["job_name"],
                    "model_name": None,  # Will be retrieved later
                    "latency": float("inf"),
                }
            )

    # Check health and update model names for backends that need checking
    now = time.time()
    checked_models = False

    for model_index in backends:
        for backend in backends[model_index]:
            backend_key = f"{backend['node']}:{backend['port']}"

            # Skip backends that aren't due for checking yet
            if (
                backend_key in backend_check_times
                and backend_check_times[backend_key] > now
            ):
                continue

            # Skip DEAD backends
            if backend_states.get(backend_key) == BackendState.DEAD:
                continue

            # Check health
            is_healthy, latency = await check_backend_health(
                backend["node"], backend["port"]
            )
            checked_models = True

            if is_healthy:
                # Update backend info
                if not backend["model_name"]:
                    # Get model name from the API
                    model_name = await get_model_name(backend["node"], backend["port"])
                    if model_name:
                        backend["model_name"] = model_name
                        model_names[model_index] = model_name

                # Update latency
                backend["latency"] = latency or float("inf")

    # Print status after changes
    if checked_models:
        print_backend_status()
    else:
        # Just print a short status update
        healthy_count = sum(
            1 for state in backend_states.values() if state == BackendState.HEALTHY
        )
        total_count = len(backend_states)
        if total_count > 0:
            print(
                colorize(
                    f"Backend status: {healthy_count}/{total_count} healthy backends",
                    Colors.GREEN if healthy_count == total_count else Colors.YELLOW,
                )
            )


def create_short_model_name(model_name):
    """Create a shortened version of the model name for display purposes."""
    if not model_name:
        return "Unknown"

    # Split by '/'
    parts = model_name.split("/")

    shortened_parts = []
    for part in parts:
        # Split by '-'
        segments = part.split("-")

        # Shorten each segment
        shortened_segments = []
        for segment in segments:
            # Take first 4 chars or all if len <= 4
            shortened_segments.append(segment[:4] if len(segment) > 4 else segment)

        # Join segments back with '-'
        shortened_parts.append("-".join(shortened_segments))

    # Join parts back with '/'
    return "/".join(shortened_parts)


def print_backend_status():
    """Print the current status of all backends."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(colorize("\n" + "=" * 100, Colors.HEADER))
    print(
        colorize(f" vLLM Backend Status at {now} ", Colors.HEADER + Colors.BOLD).center(
            100, "="
        )
    )
    print(colorize("=" * 100, Colors.HEADER))

    if not backends:
        print(colorize("No active backends found.", Colors.YELLOW))
        return

    status_table = []
    headers = [
        "Model Name",
        "Backend",
        "Job Name",
        "State",
        "Avg Latency (s)",
        "In-flight",
        "Requests",
    ]

    for model_index, backend_list in sorted(backends.items()):
        full_model_name = model_names.get(model_index, "Unknown")
        short_model_name = create_short_model_name(full_model_name)

        # Format as "model_index:short_model_name"
        display_model_name = f"{model_index}:{short_model_name}"

        for backend in backend_list:
            node = backend["node"]
            port = backend["port"]
            job_name = backend["job_name"]
            backend_key = f"{node}:{port}"

            # Get health status
            state = backend_states.get(backend_key, BackendState.HEALTHY)
            if state == BackendState.HEALTHY:
                state_str = colorize("HEALTHY", Colors.GREEN)
            elif state == BackendState.DEGRADED:
                state_str = colorize("DEGRADED", Colors.YELLOW)
            elif state == BackendState.UNHEALTHY:
                state_str = colorize("UNHEALTHY", Colors.RED)
            elif state == BackendState.INITIALIZING:
                discovery_time = backend_discovery_times.get(backend_key, time.time())
                init_time = int(time.time() - discovery_time)
                state_str = colorize(f"INITIALIZING ({init_time}s)", Colors.BLUE)
            else:  # DEAD
                state_str = colorize("DEAD", Colors.RED)

            # Get average latency
            avg_latency = get_avg_latency(node, port)
            if avg_latency == float("inf"):
                latency_str = "Unknown"
            else:
                latency_str = f"{avg_latency:.4f}"

            # Get request count and in-flight
            request_count = backend_requests.get(backend_key, 0)
            in_flight = backend_in_flight.get(backend_key, 0)

            status_table.append(
                [
                    display_model_name,
                    backend_key,
                    job_name,
                    state_str,
                    latency_str,
                    in_flight,
                    request_count,
                ]
            )

    print(tabulate(status_table, headers=headers, tablefmt="pretty"))

    # Print summary statistics
    healthy_count = sum(
        1 for state in backend_states.values() if state == BackendState.HEALTHY
    )
    degraded_count = sum(
        1 for state in backend_states.values() if state == BackendState.DEGRADED
    )
    unhealthy_count = sum(
        1 for state in backend_states.values() if state == BackendState.UNHEALTHY
    )
    initializing_count = sum(
        1 for state in backend_states.values() if state == BackendState.INITIALIZING
    )

    print(
        colorize("Summary: ", Colors.BOLD)
        + colorize(f"{healthy_count} HEALTHY", Colors.GREEN)
        + ", "
        + colorize(f"{degraded_count} DEGRADED", Colors.YELLOW)
        + ", "
        + colorize(f"{unhealthy_count} UNHEALTHY", Colors.RED)
        + ", "
        + colorize(f"{initializing_count} INITIALIZING", Colors.BLUE)
    )

    print("")


async def get_model_name(node, port):
    """Get the model name from a backend."""
    url = f"http://{node}:{port}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["id"]
    except Exception:
        pass
    return None


async def select_backend(model):
    """Select the best backend for a given model optimizing for reliability and performance."""
    # First, identify the model_index from the requested model
    target_model_index = None
    for model_index, name in model_names.items():
        if name == model:
            target_model_index = model_index
            break

    if target_model_index is None or target_model_index not in backends:
        raise HTTPException(
            status_code=400, detail=f"Model '{model}' not found in any backend"
        )

    # Filter for only HEALTHY backends
    healthy_backends = [
        b
        for b in backends[target_model_index]
        if backend_states.get(f"{b['node']}:{b['port']}", BackendState.HEALTHY)
        == BackendState.HEALTHY
    ]

    if not healthy_backends:
        # No healthy backends - try to recover any available backend for immediate use
        print(
            colorize(
                f"No healthy backends found for model '{model}', attempting recovery...",
                Colors.YELLOW,
            )
        )

        # First check if there are any backends in DEGRADED state that might be usable
        degraded_backends = [
            b
            for b in backends[target_model_index]
            if backend_states.get(f"{b['node']}:{b['port']}", BackendState.HEALTHY)
            == BackendState.DEGRADED
        ]

        if degraded_backends:
            # Use a degraded backend if available
            print(
                colorize(
                    f"Using DEGRADED backend for model '{model}' as fallback",
                    Colors.YELLOW,
                )
            )
            selected_backend = min(
                degraded_backends, key=lambda b: get_avg_latency(b["node"], b["port"])
            )
            backend_key = f"{selected_backend['node']}:{selected_backend['port']}"
            backend_requests[backend_key] = backend_requests.get(backend_key, 0) + 1
            backend_in_flight[backend_key] = backend_in_flight.get(backend_key, 0) + 1
            return selected_backend

        # Check for INITIALIZING backends
        initializing_backends = [
            b
            for b in backends[target_model_index]
            if backend_states.get(f"{b['node']}:{b['port']}", BackendState.HEALTHY)
            == BackendState.INITIALIZING
        ]

        if initializing_backends:
            # Try checking each INITIALIZING backend
            for backend in initializing_backends:
                node, port = backend["node"], backend["port"]
                backend_key = f"{node}:{port}"

                # Try to check if it's ready now
                is_healthy, _ = await check_backend_health(node, port)
                if is_healthy:
                    print(
                        colorize(
                            f"Using newly healthy backend for model '{model}'",
                            Colors.GREEN,
                        )
                    )
                    backend_requests[backend_key] = (
                        backend_requests.get(backend_key, 0) + 1
                    )
                    backend_in_flight[backend_key] = (
                        backend_in_flight.get(backend_key, 0) + 1
                    )
                    return backend

        # If no DEGRADED or ready INITIALIZING backends, try health checking UNHEALTHY ones
        for backend in backends[target_model_index]:
            node, port = backend["node"], backend["port"]
            backend_key = f"{node}:{port}"

            # Skip permanently DEAD backends
            if backend_states.get(backend_key) == BackendState.DEAD:
                continue

            # Try to recover this backend
            is_healthy, _ = await check_backend_health(node, port)
            if is_healthy or backend_states.get(backend_key) == BackendState.DEGRADED:
                healthy_backends.append(backend)
                break

    if not healthy_backends:
        raise HTTPException(
            status_code=503,
            detail="No healthy or recoverable backends available for this model",
        )

    # Sort by load and latency
    # Consider both the number of in-flight requests and the average latency
    for backend in healthy_backends:
        backend_key = f"{backend['node']}:{backend['port']}"
        backend["in_flight"] = backend_in_flight.get(backend_key, 0)

    # First prefer backends with no in-flight requests
    available_backends = [b for b in healthy_backends if b["in_flight"] == 0]

    if available_backends:
        # Sort by latency for backends with no in-flight requests
        available_backends.sort(key=lambda b: get_avg_latency(b["node"], b["port"]))

        # Add some randomness for load distribution - use weighted random choice
        weights = []
        for backend in available_backends:
            latency = get_avg_latency(backend["node"], backend["port"])
            # Convert latency to weight (lower latency = higher weight)
            if latency == float("inf"):
                weights.append(1)  # Default weight for backends with no latency data
            else:
                weights.append(1.0 / latency)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(available_backends)] * len(available_backends)
        else:
            weights = [w / total_weight for w in weights]

        selected_backend = random.choices(available_backends, weights=weights, k=1)[0]
    else:
        # If all backends have in-flight requests, select the one with the least load
        # and lowest latency using a combined score
        for backend in healthy_backends:
            latency = get_avg_latency(backend["node"], backend["port"])
            if latency == float("inf"):
                latency = 10.0  # Default high value for unknown latency
            # Calculate a score that balances load and latency
            # Lower score is better
            backend["score"] = backend["in_flight"] * 2 + latency

        # Select backend with lowest score
        selected_backend = min(healthy_backends, key=lambda b: b["score"])

    # Update request count and in-flight requests
    backend_key = f"{selected_backend['node']}:{selected_backend['port']}"
    backend_requests[backend_key] = backend_requests.get(backend_key, 0) + 1
    backend_in_flight[backend_key] = backend_in_flight.get(backend_key, 0) + 1

    return selected_backend


async def monitor_backends():
    """Periodically update backend information."""
    while True:
        try:
            await update_backend_info()
        except Exception as e:
            print(colorize(f"Error in backend monitoring: {e}", Colors.RED))
        await asyncio.sleep(MONITOR_INTERVAL)


@app.get("/v1/models")
async def list_models():
    """List all available models."""
    # Collect unique models from all backends
    available_models = []
    for model_index, name in model_names.items():
        # Check if we have at least one healthy backend for this model
        has_healthy_backend = False
        if model_index in backends:
            for backend in backends[model_index]:
                node, port = backend["node"], backend["port"]
                if (
                    backend_states.get(f"{node}:{port}", BackendState.HEALTHY)
                    == BackendState.HEALTHY
                ):
                    has_healthy_backend = True
                    break

        if has_healthy_backend and name:
            available_models.append(
                {
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "organization",
                }
            )

    return {"data": available_models, "object": "list"}


@app.post("/v1/completions")
async def create_completion(request: Request):
    """Proxy completion requests to the appropriate backend with retry logic."""
    try:
        # Get request data
        data = await request.json()

        # Extract model name
        model = data.get("model")
        if not model:
            return JSONResponse(
                content={"error": "Model name is required"}, status_code=400
            )

        # Check if streaming is requested
        stream = data.get("stream", False)

        # Set up retry logic
        retry_count = 0
        max_retries = MAX_RETRIES
        last_error = None

        while retry_count <= max_retries:
            try:
                # Select backend
                backend = await select_backend(model)
                node, port = backend["node"], backend["port"]
                backend_key = f"{node}:{port}"

                # Forward request to backend
                url = f"http://{node}:{port}/v1/completions"

                headers = {"Content-Type": "application/json"}
                for header_name, header_value in request.headers.items():
                    if header_name.lower() not in ["host", "content-length"]:
                        headers[header_name] = header_value

                if stream:
                    response = await stream_response(url, data, headers)
                    return response
                else:
                    response = await proxy_request(url, data, headers)
                    return response

            except Exception as e:
                last_error = e
                retry_count += 1

                if retry_count <= max_retries:
                    # Log retry attempt
                    print(
                        colorize(
                            f"Retry {retry_count}/{max_retries} for request to model '{model}' after error: {str(e)}",
                            Colors.YELLOW,
                        )
                    )
                    # Short delay before retry
                    await asyncio.sleep(0.5)
                else:
                    # Max retries exceeded
                    break

        # If we get here, all retries failed
        error_message = (
            f"All {max_retries} retries failed for model '{model}': {str(last_error)}"
        )
        print(colorize(error_message, Colors.RED))
        return JSONResponse(content={"error": error_message}, status_code=503)

    except HTTPException as e:
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(colorize(f"Error processing request: {e}", Colors.RED))
        return JSONResponse(
            content={"error": f"Error processing request: {str(e)}"}, status_code=500
        )


async def proxy_request(url, data, headers):
    """Forward a request to a backend and return the response with timeout handling."""
    # Extract backend key from URL for tracking
    backend_key = url.split("//")[1].split("/")[0]

    start_time = time.time()
    try:
        # Use a timeout appropriate for non-streaming requests
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)

            # Update latency for this backend
            latency = time.time() - start_time
            if backend_key in recent_latencies:
                recent_latencies[backend_key].append(latency)
                if len(recent_latencies[backend_key]) > MAX_LATENCY_SAMPLES:
                    recent_latencies[backend_key] = recent_latencies[backend_key][
                        -MAX_LATENCY_SAMPLES:
                    ]

            # Return the response from the backend
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )
    except Exception as e:
        # Mark backend as DEGRADED
        backend_failures[backend_key] = backend_failures.get(backend_key, 0) + 1
        backend_successes[backend_key] = 0

        # Update state based on failure count
        if backend_failures[backend_key] >= FAILURE_THRESHOLD:
            backend_states[backend_key] = BackendState.UNHEALTHY
            print(
                colorize(
                    f"Backend {backend_key} marked UNHEALTHY after {FAILURE_THRESHOLD} failures",
                    Colors.RED,
                )
            )
        elif backend_states.get(backend_key) == BackendState.HEALTHY:
            backend_states[backend_key] = BackendState.DEGRADED
            print(
                colorize(
                    f"Backend {backend_key} marked DEGRADED: {str(e)}", Colors.YELLOW
                )
            )

        # Update check time for next health check
        update_next_check_time(backend_key)

        print(colorize(f"Backend request failed: {e}", Colors.RED))
        raise HTTPException(status_code=503, detail=f"Backend request failed: {str(e)}")
    finally:
        # Always decrement in-flight count
        if backend_key in backend_in_flight:
            backend_in_flight[backend_key] = max(0, backend_in_flight[backend_key] - 1)


async def stream_response(url, data, headers):
    """Stream a response from a backend with timeout for first token."""
    # Extract backend key from URL for tracking
    backend_key = url.split("//")[1].split("/")[0]

    async def generate():
        nonlocal backend_key
        start_time = time.time()
        first_token_received = False

        try:
            # Use longer timeout for streaming connections but monitor first token arrival
            async with httpx.AsyncClient(timeout=None) as client:
                try:
                    async with client.stream(
                        "POST", url, json=data, headers=headers
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            # Check for first token timeout
                            if not first_token_received:
                                first_token_received = True
                                time_to_first_token = time.time() - start_time
                                if time_to_first_token > REQUEST_STREAM_TIMEOUT:
                                    # First token took too long, but we got it anyway
                                    print(
                                        colorize(
                                            f"Warning: First token from {backend_key} took {time_to_first_token:.2f}s (threshold: {REQUEST_STREAM_TIMEOUT}s)",
                                            Colors.YELLOW,
                                        )
                                    )

                            yield chunk

                    # Update latency only if we've received at least the first token
                    if first_token_received:
                        latency = time.time() - start_time
                        if backend_key in recent_latencies:
                            recent_latencies[backend_key].append(latency)
                            if len(recent_latencies[backend_key]) > MAX_LATENCY_SAMPLES:
                                recent_latencies[backend_key] = recent_latencies[
                                    backend_key
                                ][-MAX_LATENCY_SAMPLES:]
                except asyncio.TimeoutError:
                    # Handle timeout for first token
                    if (
                        not first_token_received
                        and (time.time() - start_time) > REQUEST_STREAM_TIMEOUT
                    ):
                        print(
                            colorize(
                                f"Streaming timeout: No response from {backend_key} after {REQUEST_STREAM_TIMEOUT}s",
                                Colors.RED,
                            )
                        )

                        # Mark backend as DEGRADED due to timeout
                        backend_failures[backend_key] = (
                            backend_failures.get(backend_key, 0) + 1
                        )
                        backend_successes[backend_key] = 0

                        if backend_states.get(backend_key) == BackendState.HEALTHY:
                            backend_states[backend_key] = BackendState.DEGRADED

                        # Update check time
                        update_next_check_time(backend_key)

                        # Return error message to client
                        yield json.dumps(
                            {
                                "error": f"Streaming timeout: No response after {REQUEST_STREAM_TIMEOUT}s"
                            }
                        ).encode()
                except Exception as e:
                    # Mark backend as DEGRADED
                    backend_failures[backend_key] = (
                        backend_failures.get(backend_key, 0) + 1
                    )
                    backend_successes[backend_key] = 0

                    # Update state based on failure count
                    if backend_failures[backend_key] >= FAILURE_THRESHOLD:
                        backend_states[backend_key] = BackendState.UNHEALTHY
                        print(
                            colorize(
                                f"Backend {backend_key} marked UNHEALTHY after {FAILURE_THRESHOLD} failures",
                                Colors.RED,
                            )
                        )
                    elif backend_states.get(backend_key) == BackendState.HEALTHY:
                        backend_states[backend_key] = BackendState.DEGRADED
                        print(
                            colorize(
                                f"Backend {backend_key} marked DEGRADED: {str(e)}",
                                Colors.YELLOW,
                            )
                        )

                    # Update check time
                    update_next_check_time(backend_key)

                    print(
                        colorize(f"Backend streaming request failed: {e}", Colors.RED)
                    )
                    # We can't raise an HTTPException during streaming, so we yield an error JSON
                    yield json.dumps(
                        {"error": f"Backend streaming request failed: {str(e)}"}
                    ).encode()
        finally:
            # Always ensure we decrement the in-flight count, even if client disconnects
            if backend_key in backend_in_flight:
                backend_in_flight[backend_key] = max(
                    0, backend_in_flight[backend_key] - 1
                )

    return StreamingResponse(generate())


def main():
    """Main entry point."""
    args = parse_arguments()

    # Update globals
    global MONITOR_INTERVAL
    MONITOR_INTERVAL = args.monitor_interval

    # Print startup banner
    print(colorize("=" * 100, Colors.HEADER))
    print(
        colorize(
            " vLLM API Proxy with Smart Load Balancing ", Colors.HEADER + Colors.BOLD
        ).center(100, "=")
    )
    print(colorize("=" * 100, Colors.HEADER))
    print(colorize(f"API Proxy Port: {args.port}", Colors.CYAN))
    print(
        colorize(f"Backend Monitor Interval: {MONITOR_INTERVAL} seconds", Colors.CYAN)
    )
    print(
        colorize(f"Load Balancing Algorithm: Weighted Least Response Time", Colors.CYAN)
    )
    print(colorize("=" * 100, Colors.HEADER))

    # Start the server with logging configured to suppress access logs
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="error",
        access_log=False,
    )


if __name__ == "__main__":
    main()
