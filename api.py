#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import statistics
import random
import logging

import aiohttp
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from tabulate import tabulate
import uvicorn
from pydantic import BaseModel
import httpx

# Configure logging to suppress access logs
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)


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
backend_failures = {}  # node:port -> count of recent failures
backend_requests = {}  # node:port -> count of requests sent
backend_health = {}  # node:port -> bool (is healthy)

# Monitoring settings
MONITOR_INTERVAL = 5  # seconds
MAX_LATENCY_SAMPLES = 10
FAILURE_THRESHOLD = 3
BACKEND_TIMEOUT = 10  # seconds
FAILURE_COOLDOWN = 60  # seconds after which to retry a failed backend

# Model regex pattern
MODEL_PATTERN = (
    r"^v(\d{2})(\d{2})(\d{3})$"  # Format: v[model_index][job_id][port_suffix]
)


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


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
    try:
        async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
            start_time = time.time()
            response = await client.get(url)
            latency = time.time() - start_time

            if response.status_code == 200:
                # Update latency information
                if f"{node}:{port}" not in recent_latencies:
                    recent_latencies[f"{node}:{port}"] = []

                recent_latencies[f"{node}:{port}"].append(latency)
                # Keep only the most recent samples
                if len(recent_latencies[f"{node}:{port}"]) > MAX_LATENCY_SAMPLES:
                    recent_latencies[f"{node}:{port}"] = recent_latencies[
                        f"{node}:{port}"
                    ][-MAX_LATENCY_SAMPLES:]

                # Reset failure count
                backend_failures[f"{node}:{port}"] = 0
                backend_health[f"{node}:{port}"] = True
                return True, latency
            else:
                backend_failures[f"{node}:{port}"] = (
                    backend_failures.get(f"{node}:{port}", 0) + 1
                )
                if backend_failures[f"{node}:{port}"] >= FAILURE_THRESHOLD:
                    backend_health[f"{node}:{port}"] = False
                return False, None
    except Exception as e:
        # Increment failure count
        backend_failures[f"{node}:{port}"] = (
            backend_failures.get(f"{node}:{port}", 0) + 1
        )
        if backend_failures[f"{node}:{port}"] >= FAILURE_THRESHOLD:
            backend_health[f"{node}:{port}"] = False
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
    """Update backend information by checking Slurm jobs."""
    global backends, model_names

    # Get current vLLM jobs
    jobs = await get_vllm_jobs()

    # Temporary storage for new backend info
    new_backends = {}

    # Process each job
    for job in jobs:
        model_index = job["model_index"]
        node = job["node"]
        port = job["port"]

        # Initialize if this model_index is new
        if model_index not in new_backends:
            new_backends[model_index] = []

        # Check if this backend is already in our list
        existing = False
        for backend in new_backends[model_index]:
            if backend["node"] == node and backend["port"] == port:
                existing = True
                break

        if not existing:
            # Check health and get latency
            is_healthy, latency = await check_backend_health(node, port)

            if is_healthy:
                # Get model name from the API
                model_name = await get_model_name(node, port)

                # Add to backends
                new_backends[model_index].append(
                    {
                        "node": node,
                        "port": port,
                        "job_name": job["job_name"],
                        "model_name": model_name,
                        "latency": latency or float("inf"),
                    }
                )

                # Update model name mapping
                if model_name:
                    model_names[model_index] = model_name

    # Update the global backends dictionary
    backends = new_backends

    # Print current backend status
    print_backend_status()


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
        "Model Index",
        "Model Name",
        "Backend",
        "Job Name",
        "Health",
        "Avg Latency (s)",
        "Requests",
    ]

    for model_index, backend_list in sorted(backends.items()):
        model_name = model_names.get(model_index, "Unknown")

        for backend in backend_list:
            node = backend["node"]
            port = backend["port"]
            job_name = backend["job_name"]

            # Get health status
            is_healthy = backend_health.get(f"{node}:{port}", True)
            health_status = (
                colorize("HEALTHY", Colors.GREEN)
                if is_healthy
                else colorize("UNHEALTHY", Colors.RED)
            )

            # Get average latency
            avg_latency = get_avg_latency(node, port)
            if avg_latency == float("inf"):
                latency_str = "Unknown"
            else:
                latency_str = f"{avg_latency:.4f}"

            # Get request count
            request_count = backend_requests.get(f"{node}:{port}", 0)

            status_table.append(
                [
                    model_index,
                    model_name,
                    f"{node}:{port}",
                    job_name,
                    health_status,
                    latency_str,
                    request_count,
                ]
            )

    print(tabulate(status_table, headers=headers, tablefmt="pretty"))
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
    """Select the best backend for a given model using Least Response Time algorithm."""
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

    # Filter for healthy backends
    healthy_backends = [
        b
        for b in backends[target_model_index]
        if backend_health.get(f"{b['node']}:{b['port']}", True)
    ]

    if not healthy_backends:
        # All backends are unhealthy, try to recover one
        for backend in backends[target_model_index]:
            node, port = backend["node"], backend["port"]
            is_healthy, _ = await check_backend_health(node, port)
            if is_healthy:
                healthy_backends.append(backend)
                break

    if not healthy_backends:
        raise HTTPException(
            status_code=503, detail="No healthy backends available for this model"
        )

    # Sort by latency (Least Response Time algorithm)
    healthy_backends.sort(key=lambda b: get_avg_latency(b["node"], b["port"]))

    # Add some randomness to avoid all traffic going to a single backend
    # Use a weighted random choice based on latency
    weights = []
    for backend in healthy_backends:
        latency = get_avg_latency(backend["node"], backend["port"])
        # Convert latency to weight (lower latency = higher weight)
        if latency == float("inf"):
            weights.append(1)  # Default weight for backends with no latency data
        else:
            weights.append(1.0 / latency)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        # If all weights are zero, use equal weights
        weights = [1.0 / len(healthy_backends)] * len(healthy_backends)
    else:
        weights = [w / total_weight for w in weights]

    # Select backend using weighted random choice
    selected_backend = random.choices(healthy_backends, weights=weights, k=1)[0]

    # Update request count
    backend_key = f"{selected_backend['node']}:{selected_backend['port']}"
    backend_requests[backend_key] = backend_requests.get(backend_key, 0) + 1

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
                if backend_health.get(f"{node}:{port}", True):
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
    """Proxy completion requests to the appropriate backend."""
    try:
        # Get request data
        data = await request.json()

        # Extract model name
        model = data.get("model")
        if not model:
            return JSONResponse(
                content={"error": "Model name is required"}, status_code=400
            )

        # Select backend
        backend = await select_backend(model)
        node, port = backend["node"], backend["port"]

        # Check if streaming is requested
        stream = data.get("stream", False)

        # Forward request to backend
        url = f"http://{node}:{port}/v1/completions"

        headers = {"Content-Type": "application/json"}
        for header_name, header_value in request.headers.items():
            if header_name.lower() not in ["host", "content-length"]:
                headers[header_name] = header_value

        if stream:
            return await stream_response(url, data, headers)
        else:
            return await proxy_request(url, data, headers)

    except HTTPException as e:
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(colorize(f"Error processing request: {e}", Colors.RED))
        return JSONResponse(
            content={"error": f"Error processing request: {str(e)}"}, status_code=500
        )


async def proxy_request(url, data, headers):
    """Forward a request to a backend and return the response."""
    start_time = time.time()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=data, headers=headers)

            # Update latency for this backend
            node_port = url.split("//")[1].split("/")[0]  # Extract node:port from URL
            latency = time.time() - start_time

            if node_port in recent_latencies:
                recent_latencies[node_port].append(latency)
                if len(recent_latencies[node_port]) > MAX_LATENCY_SAMPLES:
                    recent_latencies[node_port] = recent_latencies[node_port][
                        -MAX_LATENCY_SAMPLES:
                    ]

            # Return the response from the backend
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        except Exception as e:
            # Mark backend as unhealthy
            node_port = url.split("//")[1].split("/")[0]
            backend_failures[node_port] = backend_failures.get(node_port, 0) + 1
            if backend_failures[node_port] >= FAILURE_THRESHOLD:
                backend_health[node_port] = False

            print(colorize(f"Backend request failed: {e}", Colors.RED))
            raise HTTPException(
                status_code=503, detail=f"Backend request failed: {str(e)}"
            )


async def stream_response(url, data, headers):
    """Stream a response from a backend."""

    async def generate():
        start_time = time.time()
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                async with client.stream(
                    "POST", url, json=data, headers=headers
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

                # Update latency after streaming completes
                node_port = url.split("//")[1].split("/")[0]
                latency = time.time() - start_time

                if node_port in recent_latencies:
                    recent_latencies[node_port].append(latency)
                    if len(recent_latencies[node_port]) > MAX_LATENCY_SAMPLES:
                        recent_latencies[node_port] = recent_latencies[node_port][
                            -MAX_LATENCY_SAMPLES:
                        ]

            except Exception as e:
                # Mark backend as unhealthy
                node_port = url.split("//")[1].split("/")[0]
                backend_failures[node_port] = backend_failures.get(node_port, 0) + 1
                if backend_failures[node_port] >= FAILURE_THRESHOLD:
                    backend_health[node_port] = False

                print(colorize(f"Backend streaming request failed: {e}", Colors.RED))
                # We can't raise an HTTPException during streaming, so we yield an error JSON
                yield json.dumps(
                    {"error": f"Backend streaming request failed: {str(e)}"}
                ).encode()

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
        log_level="error",  # Only show error logs, not info or access logs
        access_log=False,  # Disable access logs completely
    )


if __name__ == "__main__":
    main()
