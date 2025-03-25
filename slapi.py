#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import tempfile
import statistics
import random
import logging
import yaml
from enum import Enum, auto
import hashlib
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from tabulate import tabulate
import uvicorn
import httpx

# Configure logging to suppress access logs
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

# Test mode configuration
TEST_BASE_TIME = 180  # Initial base time (seconds)
TEST_INTERVAL = 45  # Time interval between job failures (seconds)

# Time limits for non-test mode (in seconds)
TIME_LIMIT_LOWER = 1800  # 30 minutes
TIME_LIMIT_UPPER = 3600  # 60 minutes

# Track job submission counts for test mode
test_job_submission_counts = {}  # {(model_index, job_id): count}


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


# Shared global state
config = {}  # Loaded from config.yaml
backends = {}  # model_index -> list of backend info
model_names = {}  # model_index -> model_name
recent_latencies = {}  # node:port -> list of recent latencies
backend_failures = {}  # node:port -> count of consecutive failures
health_check_failures = {}  # node:port -> count of consecutive health check failures
backend_successes = {}  # node:port -> count of consecutive successful health checks
backend_requests = {}  # node:port -> count of requests sent
backend_states = {}  # node:port -> BackendState
backend_in_flight = {}  # node:port -> count of in-flight requests
backend_check_times = {}  # node:port -> next check time
backend_discovery_times = {}  # node:port -> when backend was first discovered

# For job balancing
model_request_counts = {}  # model_index -> total requests across all backends
model_jobs_target = {}  # model_index -> target number of jobs
model_jobs_current = {}  # model_index -> current number of jobs
total_number_of_jobs = 0  # Total jobs across all models

# FastAPI app
app = None

# Monitoring settings
MONITOR_INTERVAL = 5  # seconds
MAX_LATENCY_SAMPLES = 10
FAILURE_THRESHOLD = 10  # consecutive failures before marking as UNHEALTHY
SUCCESS_THRESHOLD = 3  # consecutive successes before promoting from DEGRADED to HEALTHY
BACKEND_TIMEOUT = 10  # seconds for health check timeout
INITIAL_GRACE_PERIOD = 0  # No grace period - backends should be checked immediately
MAX_INITIALIZING_TIME = 300  # seconds before INITIALIZING -> UNHEALTHY
REQUEST_STREAM_TIMEOUT = 60  # seconds to wait for first token in streaming response
MAX_RETRIES = 3  # maximum number of retries for a failed request
DEGRADED_CHECK_INTERVAL = 3  # seconds between DEGRADED backend checks
UNHEALTHY_CHECK_INTERVAL = 10  # initial seconds between UNHEALTHY backend checks
INITIALIZING_CHECK_INTERVAL = 3  # seconds between INITIALIZING backend checks
MAX_BACKOFF_INTERVAL = 60  # maximum seconds for exponential backoff

# Log deduplication settings and storage
ERROR_LOG_DEDUP_WINDOW = 60  # seconds to suppress duplicate error logs
error_log_cache = {}  # message_hash -> timestamp


def colorize(text, color):
    """Add color to text."""
    return f"{color}{text}{Colors.ENDC}"


def log_once(message, color=None):
    """Log a message only once within the deduplication window."""
    # Create a hash of the message to use as key
    msg_hash = hashlib.md5(message.encode()).hexdigest()

    current_time = time.time()

    # Clean up old entries
    for key in list(error_log_cache.keys()):
        if current_time - error_log_cache[key] > ERROR_LOG_DEDUP_WINDOW:
            del error_log_cache[key]

    # Check if this message was logged recently
    if msg_hash in error_log_cache:
        return False  # Already logged recently

    # Log the message and update cache
    error_log_cache[msg_hash] = current_time
    if color:
        print(colorize(message, color))
    else:
        print(message)
    return True


# Main functions and application logic will be added below


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=colorize(
            "vLLM Management System with Smart Load Balancing", Colors.HEADER
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--port", type=int, default=9090, help="Port to run the API proxy server on"
    )
    parser.add_argument(
        "--job-prefix",
        type=str,
        default="v",
        help="Single lowercase letter prefix for job names (default: 'v')",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=5,
        help="Interval in seconds between job status and backend health checks",
    )
    parser.add_argument(
        "--cancel-all", action="store_true", help="Cancel all vLLM jobs and exit"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with simulated job failures at regular intervals",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run only the API proxy without job management",
    )
    parser.add_argument(
        "--job-only",
        action="store_true",
        help="Run only the job management without API proxy",
    )

    args = parser.parse_args()

    # Validate job prefix is a single lowercase English letter
    if not (
        len(args.job_prefix) == 1
        and args.job_prefix.islower()
        and args.job_prefix.isalpha()
    ):
        print(
            colorize(
                f"Error: job prefix must be a single lowercase English letter, got '{args.job_prefix}'",
                Colors.RED,
            )
        )
        sys.exit(1)

    return args


def load_config(config_path):
    """Load configuration from YAML file and process it for dynamic job balancing."""
    global config, total_number_of_jobs, model_jobs_target, model_jobs_current

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(colorize(f"Error loading configuration: {e}", Colors.RED))
        sys.exit(1)

    # Extract total number of jobs if provided
    if "total_number_of_jobs" in config:
        total_number_of_jobs = config["total_number_of_jobs"]

        # Count the number of valid model entries
        model_indices = [k for k in config if isinstance(k, int)]
        model_count = len(model_indices)

        if model_count == 0:
            print(colorize("No valid model entries found in config", Colors.RED))
            sys.exit(1)

        # Check if we have enough jobs for minimum 2 per model
        if total_number_of_jobs < 2 * model_count:
            print(
                colorize(
                    f"Error: total_number_of_jobs ({total_number_of_jobs}) must be at least 2 times the number of models ({model_count})",
                    Colors.RED,
                )
            )
            print(colorize(f"Need at least {2 * model_count} total jobs", Colors.RED))
            sys.exit(1)

        # Distribute jobs evenly initially
        base_jobs_per_model = total_number_of_jobs // model_count
        extra_jobs = total_number_of_jobs % model_count

        print(
            colorize(
                f"Distributing {total_number_of_jobs} total jobs among {model_count} models: "
                f"{base_jobs_per_model} each with {extra_jobs} extra",
                Colors.CYAN,
            )
        )

        # Assign initial target jobs per model
        for i, model_index in enumerate(sorted(model_indices)):
            if i < extra_jobs:
                model_jobs_target[model_index] = base_jobs_per_model + 1
            else:
                model_jobs_target[model_index] = base_jobs_per_model

            # Initialize current job count
            model_jobs_current[model_index] = 0
            # Initialize request count
            model_request_counts[model_index] = 0
    else:
        # Legacy mode: use specified number_of_jobs per model
        total_number_of_jobs = 0
        for model_index, model_config in config.items():
            if isinstance(model_index, int) and "number_of_jobs" in model_config:
                model_jobs_target[model_index] = model_config["number_of_jobs"]
                model_jobs_current[model_index] = 0
                model_request_counts[model_index] = 0
                total_number_of_jobs += model_config["number_of_jobs"]


def ensure_log_dir():
    """Ensure the log directory exists."""
    log_dir = Path("./vllm_logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def extract_job_info(job_name):
    """Extract model index, job ID and port from job name."""
    try:
        # Format is {prefix}{model_index:02d}{job_id:02d}{port_suffix}
        # Where prefix is a single character and port_suffix is the last 3 digits of the port number
        model_index = int(job_name[1:3])
        job_id = int(job_name[3:5])

        # Extract port suffix - it should be the last 3 characters of the base name
        # We need to handle both standard port suffixes and potential additional suffixes
        port_suffix = job_name[5:]
        if len(port_suffix) > 3:
            port_suffix = port_suffix[:3]  # Take only first 3 digits

        port = int("9" + port_suffix)  # Add "9" prefix to make it 9XXX
        return model_index, job_id, port
    except (ValueError, IndexError) as e:
        # Optional debugging
        # print(f"Failed to extract job info from {job_name}: {e}")
        return None, None, None


def get_available_port(used_ports):
    """Get a port number that is not in the used_ports list."""
    excluded_ports = {9000, 9090, 9999}
    potential_ports = [p for p in range(9000, 10000) if p not in excluded_ports]
    random.shuffle(potential_ports)

    for port in potential_ports:
        if port not in used_ports:
            return port

    return random.randint(9000, 9999)


def get_used_ports(username, job_prefix="v"):
    """Get a list of ports used by existing vLLM jobs."""
    used_ports = []
    cmd = ["squeue", "-u", username, "-h", "-o", "%j"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for job_name in result.stdout.strip().split("\n"):
        if job_name and job_name.startswith(job_prefix):
            try:
                port_suffix = job_name[-3:]
                used_ports.append(int("9" + port_suffix))
            except (ValueError, IndexError):
                pass

    return used_ports


def choose_model_for_new_job():
    """Dynamically choose which model to launch a new job for based on request patterns."""
    global model_request_counts, model_jobs_current, model_jobs_target

    # If no requests yet, distribute according to initial allocation
    total_requests = sum(model_request_counts.values())
    if total_requests == 0:
        # Find model with biggest gap between target and current
        max_gap = -1
        chosen_model = None

        for model_index in model_jobs_target:
            gap = model_jobs_target[model_index] - model_jobs_current[model_index]
            if gap > max_gap and gap > 0:  # Only consider models that need more jobs
                max_gap = gap
                chosen_model = model_index

        return chosen_model

    # Calculate ideal job distribution based on request distribution
    ideal_jobs = {}
    for model_index in model_request_counts:
        if model_index in model_jobs_target:  # Make sure model is in our config
            request_ratio = model_request_counts[model_index] / total_requests
            ideal_jobs[model_index] = int(total_number_of_jobs * request_ratio)

    # Ensure minimum of 2 jobs per model
    for model_index in ideal_jobs:
        if ideal_jobs[model_index] < 2:
            ideal_jobs[model_index] = 2

    # Adjust to match total_number_of_jobs
    total_ideal = sum(ideal_jobs.values())
    if total_ideal != total_number_of_jobs:
        # Distribute excess or deficit
        diff = total_number_of_jobs - total_ideal
        sorted_models = sorted(
            ideal_jobs.keys(), key=lambda m: model_request_counts[m], reverse=(diff > 0)
        )

        # Distribute the difference
        for i in range(abs(diff)):
            model_index = sorted_models[i % len(sorted_models)]
            ideal_jobs[model_index] += 1 if diff > 0 else -1

            # Ensure minimum of 2 jobs per model
            if ideal_jobs[model_index] < 2:
                ideal_jobs[model_index] = 2

    # Update target job counts
    model_jobs_target.update(ideal_jobs)

    # Find model with biggest gap between target and current
    max_gap = -1
    chosen_model = None

    for model_index, target in model_jobs_target.items():
        current = model_jobs_current.get(model_index, 0)
        gap = target - current
        if gap > max_gap and gap > 0:  # Only consider models that need more jobs
            max_gap = gap
            chosen_model = model_index

    return chosen_model


def get_next_job_id(model_index, job_prefix="v"):
    """Get the next available job ID for a model."""
    # Find all currently used job IDs for this model
    used_job_ids = set()
    username = os.environ.get("USER", subprocess.getoutput("whoami"))
    cmd = ["squeue", "-u", username, "-h", "-o", "%j"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for job_name in result.stdout.strip().split("\n"):
        if job_name and job_name.startswith(job_prefix):
            job_model_index, job_id, _ = extract_job_info(job_name)
            if job_model_index == model_index and job_id is not None:
                used_job_ids.add(job_id)

    # Find the first unused job ID starting from 1
    job_id = 1
    while job_id in used_job_ids:
        job_id += 1

    return job_id


def submit_job(
    model_index,
    job_id,
    model_name,
    gpu,
    current_dir,
    username,
    test_mode=False,
    job_prefix="v",
):
    """Submit a vLLM job to Slurm."""
    global model_jobs_current

    # Slurm job parameters
    slurm_params = {
        "qos": "scavenger",
        "time_limit": "01:00:00",  # This will be overridden
        "cpu": "4",
        "mem": "16G",
        "partition": "--partition=scavenger",
        "account": "--account=scavenger",
    }

    # Handle test mode time limit
    if test_mode:
        key = (model_index, job_id)
        # Initialize or increment submission count
        if key not in test_job_submission_counts:
            test_job_submission_counts[key] = 0
        else:
            test_job_submission_counts[key] += 1

        submission_count = test_job_submission_counts[key]

        # Use the number of jobs in the model or default to a reasonable value
        num_jobs = model_jobs_target.get(model_index, 6)

        if submission_count == 0:
            # For first submission, stagger time limits based on job_id
            # Job 1: BASE_TIME, Job 2: BASE_TIME + INTERVAL, etc.
            time_limit_seconds = TEST_BASE_TIME + ((job_id - 1) * TEST_INTERVAL)
        else:
            # For resubmissions, always use a fixed time interval
            time_limit_seconds = TEST_INTERVAL * num_jobs
    else:
        # In non-test mode, use a random time between TIME_LIMIT_LOWER and TIME_LIMIT_UPPER
        time_limit_seconds = random.randint(TIME_LIMIT_LOWER, TIME_LIMIT_UPPER)

    # Format time as HH:MM:SS
    minutes, seconds = divmod(time_limit_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    slurm_params["time_limit"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    used_ports = get_used_ports(username, job_prefix)
    port = get_available_port(used_ports)
    port_suffix = str(port)[-3:]

    formatted_model_index = f"{model_index:02d}"
    formatted_job_id = f"{job_id:02d}"
    job_name = f"{job_prefix}{formatted_model_index}{formatted_job_id}{port_suffix}"

    log_dir = ensure_log_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create job script
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file_path = temp_file.name

        job_script = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --qos={slurm_params['qos']}
#SBATCH {slurm_params['partition']}
#SBATCH {slurm_params['account']}
#SBATCH --time={slurm_params['time_limit']}
#SBATCH --gres={gpu}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm_params['cpu']}
#SBATCH --mem={slurm_params['mem']}
#SBATCH --output={log_dir}/{timestamp}_{job_name}_{model_name.replace('/', '_')}_%N.log
#SBATCH --error={log_dir}/{timestamp}_{job_name}_{model_name.replace('/', '_')}_%N.log

# Run the vLLM server
cd {current_dir}
source ./venv/bin/activate

echo "========================================================"
echo "Starting vLLM server"
echo "Job Name: {job_name}"
echo "Model: {model_name}"
echo "Port: {port}"
echo "GPU: {gpu}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Time Limit: {slurm_params['time_limit']}"
echo "========================================================"

vllm serve {model_name} --port {port}
"""
        temp_file.write(job_script)

    print(
        colorize(
            f"Submitting vLLM job {job_name} for model {model_name}...", Colors.CYAN
        )
    )
    result = subprocess.run(["sbatch", temp_file_path], capture_output=True, text=True)

    os.unlink(temp_file_path)

    if result.returncode == 0:
        print(colorize(f"Job submitted: {result.stdout.strip()}", Colors.GREEN))
        if test_mode:
            print(
                colorize(
                    f"  [TEST MODE] Time limit: {slurm_params['time_limit']} (Submission #{test_job_submission_counts.get((model_index, job_id), 0)+1})",
                    Colors.YELLOW,
                )
            )
        else:
            print(
                colorize(
                    f"  Time limit: {slurm_params['time_limit']}",
                    Colors.GREEN,
                )
            )

        # Update the job count for this model
        model_jobs_current[model_index] = model_jobs_current.get(model_index, 0) + 1

        return True, job_name, port, model_name
    else:
        print(colorize(f"Error submitting job: {result.stderr}", Colors.RED))
        return False, job_name, port, model_name


def get_job_id_by_name(job_name, username):
    """Get the Slurm job ID for a job with the given name."""
    cmd = ["squeue", "-u", username, "-n", job_name, "-h", "-o", "%i"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() or None


def get_job_status(job_name, username):
    """Get detailed status of a job."""
    cmd = ["squeue", "-u", username, "-n", job_name, "-h", "-o", "%i|%j|%T|%N|%M|%L"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if not result.stdout.strip():
        return None

    try:
        parts = result.stdout.strip().split("|")
        if len(parts) >= 6:
            return {
                "job_id": parts[0],
                "job_name": parts[1],
                "state": parts[2],
                "node": parts[3],
                "time": parts[4],
                "time_limit": parts[5],
            }
    except Exception as e:
        print(colorize(f"Error parsing job status: {e}", Colors.RED))

    return None


def job_exists(job_name, username):
    """Check if a job exists (either running or pending)."""
    cmd = ["squeue", "-u", username, "-n", job_name, "-h"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip() or get_job_id_by_name(job_name, username))


def get_all_vllm_jobs(username, job_prefix="v"):
    """Get all vLLM jobs for the user."""
    cmd = ["squeue", "-u", username, "-h", "-o", "%j"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return [
        job_name
        for job_name in result.stdout.strip().split("\n")
        if job_name and job_name.startswith(job_prefix)
    ]


def cancel_all_vllm_jobs(username, job_prefix="v"):
    """Cancel all vLLM jobs for the user."""
    print(colorize("\nCancelling all vLLM jobs...", Colors.YELLOW))

    current_jobs = get_all_vllm_jobs(username, job_prefix)

    if not current_jobs:
        print(colorize("No vLLM jobs found to cancel.", Colors.CYAN))
        return

    print(colorize(f"Found {len(current_jobs)} vLLM jobs to cancel:", Colors.CYAN))
    for job_name in current_jobs:
        print(colorize(f"  - {job_name}", Colors.CYAN))

    # Cancel each job individually by job ID (more reliable approach)
    cancelled_count = 0
    for job_name in current_jobs:
        job_id = get_job_id_by_name(job_name, username)
        if job_id:
            subprocess.run(["scancel", job_id], capture_output=True)
            print(colorize(f"Cancelled job {job_name} (ID: {job_id})", Colors.GREEN))
            cancelled_count += 1

    if cancelled_count == len(current_jobs):
        print(colorize("All jobs have been successfully cancelled.", Colors.GREEN))
    else:
        print(
            colorize(
                f"Warning: Only cancelled {cancelled_count}/{len(current_jobs)} jobs.",
                Colors.YELLOW,
            )
        )

        # Verify cancellation
        time.sleep(2)
        remaining_jobs = get_all_vllm_jobs(username)

        if remaining_jobs:
            print(
                colorize(
                    f"Attempting second pass for {len(remaining_jobs)} remaining jobs...",
                    Colors.YELLOW,
                )
            )
            # Try wildcard approach as fallback
            cmd = ["scancel", "-n", "v*", "-u", username]
            subprocess.run(cmd, capture_output=True)

            # Final verification
            time.sleep(1)
            final_jobs = get_all_vllm_jobs(username)
            if final_jobs:
                print(
                    colorize(
                        f"Warning: {len(final_jobs)} jobs still remain after second attempt.",
                        Colors.RED,
                    )
                )
                for job_name in final_jobs:
                    print(colorize(f"  - {job_name}", Colors.RED))
            else:
                print(
                    colorize(
                        "All jobs have been successfully cancelled on second attempt.",
                        Colors.GREEN,
                    )
                )
        else:
            print(colorize("All jobs have been successfully cancelled.", Colors.GREEN))


def get_short_model_name(model_name):
    """Create a short, readable version of model name."""
    if not model_name:
        return model_name

    # First split by '/'
    parts = []
    for part in model_name.split("/"):
        # Then split each part by '-'
        if "-" in part:
            subparts = []
            for subpart in part.split("-"):
                # Truncate each subpart to 4 chars if longer
                if len(subpart) > 4:
                    subparts.append(subpart[:4])
                else:
                    subparts.append(subpart)
            # Rejoin with original hyphens
            parts.append("-".join(subparts))
        else:
            # Handle parts with no hyphens
            if len(part) > 4:
                parts.append(part[:4])
            else:
                parts.append(part)

    # Rejoin with original slashes
    return "/".join(parts)


# Backend health monitoring functions
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


def get_avg_latency(node, port):
    """Get the average latency for a backend."""
    if f"{node}:{port}" in recent_latencies and recent_latencies[f"{node}:{port}"]:
        return statistics.mean(recent_latencies[f"{node}:{port}"])
    return float("inf")  # Return infinity if no latency data is available


def update_next_check_time(backend_key):
    """Update when to next check a backend based on its current state."""
    current_state = backend_states.get(backend_key, BackendState.HEALTHY)

    if current_state == BackendState.DEGRADED:
        # Check DEGRADED backends every 3 seconds
        backend_check_times[backend_key] = time.time() + DEGRADED_CHECK_INTERVAL

    elif current_state == BackendState.UNHEALTHY:
        # Use exponential backoff for UNHEALTHY backends, with a maximum interval
        failures = health_check_failures.get(backend_key, 0)
        backoff = min(
            UNHEALTHY_CHECK_INTERVAL * (2 ** (failures - FAILURE_THRESHOLD)),
            MAX_BACKOFF_INTERVAL,
        )
        backend_check_times[backend_key] = time.time() + backoff

    elif current_state == BackendState.INITIALIZING:
        # Check INITIALIZING backends every 3 seconds
        backend_check_times[backend_key] = time.time() + INITIALIZING_CHECK_INTERVAL


def handle_health_check_failure(backend_key, error_msg):
    """Handle a failed health check and update backend state accordingly."""
    # Increment health check failure count and reset success count
    health_check_failures[backend_key] = health_check_failures.get(backend_key, 0) + 1
    backend_successes[backend_key] = 0

    current_state = backend_states.get(backend_key, BackendState.HEALTHY)

    # State transitions based on failures
    if current_state == BackendState.HEALTHY:
        if health_check_failures[backend_key] == 1:
            # First failure for a healthy backend -> DEGRADED
            backend_states[backend_key] = BackendState.DEGRADED
            log_once(
                f"Backend {backend_key} degraded: HEALTHY → DEGRADED (health check: {error_msg})",
                Colors.YELLOW,
            )

    elif current_state == BackendState.DEGRADED:
        if health_check_failures[backend_key] >= FAILURE_THRESHOLD:
            # Too many failures for a degraded backend -> UNHEALTHY
            # This transition only happens with dedicated health check failures, not payload failures
            backend_states[backend_key] = BackendState.UNHEALTHY
            log_once(
                f"Backend {backend_key} unhealthy: DEGRADED → UNHEALTHY after {health_check_failures[backend_key]} health check failures",
                Colors.RED,
            )

    elif current_state == BackendState.INITIALIZING:
        # Check if we've been trying to initialize for too long
        discovery_time = backend_discovery_times.get(backend_key, time.time())
        initializing_time = time.time() - discovery_time

        if initializing_time > INITIAL_GRACE_PERIOD + MAX_INITIALIZING_TIME:
            # Backend failed to initialize within time limit
            backend_states[backend_key] = BackendState.UNHEALTHY
            log_once(
                f"Backend {backend_key} failed to initialize after {int(initializing_time)}s: INITIALIZING → UNHEALTHY",
                Colors.RED,
            )

    # Update check time for next health check
    update_next_check_time(backend_key)

    return False, None


def handle_payload_failure(backend_key, error_msg):
    """Handle a failed payload request and update backend state accordingly."""
    # Increment payload failure count
    backend_failures[backend_key] = backend_failures.get(backend_key, 0) + 1

    current_state = backend_states.get(backend_key, BackendState.HEALTHY)

    # For payload failures, we only do the HEALTHY → DEGRADED transition
    # We never go straight to UNHEALTHY from payload failures
    if current_state == BackendState.HEALTHY:
        # First payload failure for a healthy backend -> immediately DEGRADED
        backend_states[backend_key] = BackendState.DEGRADED
        log_once(
            f"Backend {backend_key} degraded: HEALTHY → DEGRADED (payload request: {error_msg})",
            Colors.YELLOW,
        )

        # Reset health check failures counter - give health check system a fresh start
        health_check_failures[backend_key] = 0

        # Update check time for next health check
        update_next_check_time(backend_key)


async def check_backend_health(node, port, force_check=False):
    """Check if a backend is healthy by making a simple request."""
    url = f"http://{node}:{port}/v1/models"
    backend_key = f"{node}:{port}"

    # Skip if not yet due for checking and not forced
    if not force_check and backend_key in backend_check_times:
        if time.time() < backend_check_times[backend_key]:
            # Return the last known state as a best guess
            if backend_states.get(backend_key) == BackendState.HEALTHY:
                return True, get_avg_latency(node, port)
            return False, None

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

                # Update health tracking - reset BOTH failure counters on success
                backend_failures[backend_key] = 0
                health_check_failures[backend_key] = 0
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
                    log_once(
                        f"Backend {backend_key} recovered: DEGRADED → HEALTHY",
                        Colors.GREEN,
                    )

                elif (
                    current_state == BackendState.UNHEALTHY
                    and backend_successes[backend_key] >= SUCCESS_THRESHOLD
                ):
                    # Promote from UNHEALTHY to DEGRADED after consecutive successes
                    backend_states[backend_key] = BackendState.DEGRADED
                    log_once(
                        f"Backend {backend_key} improving: UNHEALTHY → DEGRADED",
                        Colors.YELLOW,
                    )

                elif current_state == BackendState.INITIALIZING:
                    # New backend is now healthy
                    backend_states[backend_key] = BackendState.HEALTHY
                    log_once(
                        f"Backend {backend_key} ready: INITIALIZING → HEALTHY",
                        Colors.GREEN,
                    )

                # Update check time for next health check based on state
                update_next_check_time(backend_key)

                return True, latency
            else:
                # Handle health check failure (not payload failure)
                return handle_health_check_failure(
                    backend_key, f"HTTP {response.status_code}"
                )
    except Exception as e:
        # Handle health check exception (not payload exception)
        return handle_health_check_failure(backend_key, str(e))


async def get_vllm_jobs(job_prefix="v"):
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

            if job_name.startswith(job_prefix):
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
        # Handle case where node isn't assigned yet (pending jobs)
        elif len(parts) == 1 and parts[0].startswith(job_prefix):
            job_name = parts[0]
            model_index, job_id, port = extract_job_info(job_name)
            if model_index is not None:
                jobs.append(
                    {
                        "job_name": job_name,
                        "node": "pending",  # Use 'pending' as placeholder
                        "model_index": model_index,
                        "job_id": job_id,
                        "port": port,
                    }
                )

    return jobs


def initialize_new_backend(model_index, node, port, job_name):
    """Initialize a newly discovered backend correctly."""
    backend_key = f"{node}:{port}"

    # Record discovery time
    backend_discovery_times[backend_key] = time.time()

    # Mark as INITIALIZING
    backend_states[backend_key] = BackendState.INITIALIZING
    backend_failures[backend_key] = 0
    health_check_failures[backend_key] = 0
    backend_successes[backend_key] = 0

    # Set check time to now - check immediately for first time
    backend_check_times[backend_key] = time.time()

    # Return the new backend object
    return {
        "node": node,
        "port": port,
        "job_name": job_name,
        "model_name": None,  # Will be retrieved later
        "latency": float("inf"),
    }


async def update_backend_info(job_prefix="v"):
    """Update backend information by checking Slurm jobs and managing backend lifecycle."""
    global backends, model_names, model_jobs_current

    # Reset current job counts for accurate tracking
    for model_index in model_jobs_current:
        model_jobs_current[model_index] = 0

    # Get current vLLM jobs from Slurm
    jobs = await get_vllm_jobs(job_prefix)

    # Create a set of all current backend keys from Slurm
    current_backend_keys = set()
    for job in jobs:
        # Increment job count for this model (both pending and running jobs)
        model_index = job["model_index"]
        model_jobs_current[model_index] = model_jobs_current.get(model_index, 0) + 1

        # Only add to backend keys if it's not a pending job
        if job["node"] != "pending":
            current_backend_keys.add(f"{job['node']}:{job['port']}")

    # Identify missing backends (removed from Slurm output)
    for model_index in list(backends.keys()):
        for backend in list(backends[model_index]):
            backend_key = f"{backend['node']}:{backend['port']}"

            if backend_key not in current_backend_keys:
                # Backend no longer in Slurm output - mark as DEAD and remove
                log_once(
                    f"Backend {backend_key} no longer in Slurm output - marked as DEAD",
                    Colors.RED,
                )
                backends[model_index].remove(backend)
                backend_states[backend_key] = BackendState.DEAD

                # Clean up associated resources
                if backend_key in recent_latencies:
                    del recent_latencies[backend_key]
                if backend_key in backend_failures:
                    del backend_failures[backend_key]
                if backend_key in health_check_failures:
                    del health_check_failures[backend_key]
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

        # Skip pending jobs for backend creation
        if node == "pending":
            continue

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
            log_once(
                f"Discovered new backend {backend_key} for model index {model_index}",
                Colors.CYAN,
            )

            # Add to backends with proper initialization
            backends[model_index].append(
                initialize_new_backend(model_index, node, port, job["job_name"])
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

            # Check health - this is a health check, not a payload request
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

    return checked_models


def print_job_status(job_prefix="v"):
    """Print the status of all Slurm jobs."""
    username = os.environ.get("USER", subprocess.getoutput("whoami"))
    current_jobs = get_all_vllm_jobs(username, job_prefix)

    job_status_table = []
    job_status_headers = [
        "Model",
        "Job",
        "Status",
        "Node",
        "Runtime",
        "Timeout",
        "Port",
        "Resubmits",
    ]

    for job_name in current_jobs:
        model_index, job_id, port = extract_job_info(job_name)
        if model_index is None:
            continue

        job_status = get_job_status(job_name, username)
        model_name = model_names.get(model_index, f"Model-{model_index}")
        short_model = get_short_model_name(model_name)

        # Get resubmit count
        key = (model_index, job_id)
        resubmit_count = test_job_submission_counts.get(key, 0)

        if job_status:
            state = job_status["state"]
            state_color = Colors.GREEN if state == "RUNNING" else Colors.YELLOW

            status_row = [
                f"{model_index}:{short_model}",
                job_id,
                colorize(state, state_color),
                job_status["node"],
                job_status["time"],
                job_status["time_limit"],
                port,
                resubmit_count,
            ]
        else:
            status_row = [
                f"{model_index}:{short_model}",
                job_id,
                colorize("UNKNOWN", Colors.YELLOW),
                "N/A",
                "N/A",
                "N/A",
                port,
                resubmit_count,
            ]

        job_status_table.append(status_row)

    # Sort by model index and job id
    job_status_table.sort(key=lambda x: (int(x[0].split(":")[0]), x[1]))

    if job_status_table:
        print(colorize("\nJob Status:", Colors.HEADER))
        print(tabulate(job_status_table, headers=job_status_headers, tablefmt="pretty"))
    else:
        print(colorize("\nNo vLLM jobs currently running.", Colors.YELLOW))


def print_backend_status():
    """Print the status of all backends."""
    if not backends:
        print(colorize("\nNo active backends found.", Colors.YELLOW))
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
        short_model_name = get_short_model_name(full_model_name)

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

    if status_table:
        print(colorize("\nBackend Status:", Colors.HEADER))
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


def print_model_balance_status():
    """Print the current job and request balance between models."""
    balance_table = []

    if "total_number_of_jobs" in config:
        # For dynamic balancing mode with request tracking
        total_requests = sum(model_request_counts.values())
        total_jobs = sum(model_jobs_current.values())

        # Update target jobs based on current request distribution
        if total_requests > 0:
            # Calculate ideal job distribution based on request distribution
            ideal_jobs = {}
            for model_index in model_request_counts:
                if model_index in model_jobs_target:  # Make sure model is in our config
                    request_ratio = model_request_counts[model_index] / total_requests
                    ideal_jobs[model_index] = int(total_number_of_jobs * request_ratio)

            # Ensure minimum of 2 jobs per model
            for model_index in ideal_jobs:
                if ideal_jobs[model_index] < 2:
                    ideal_jobs[model_index] = 2

            # Adjust to match total_number_of_jobs
            total_ideal = sum(ideal_jobs.values())
            if total_ideal != total_number_of_jobs:
                # Distribute excess or deficit
                diff = total_number_of_jobs - total_ideal
                sorted_models = sorted(
                    ideal_jobs.keys(),
                    key=lambda m: model_request_counts[m],
                    reverse=(diff > 0),
                )

                # Distribute the difference
                for i in range(abs(diff)):
                    model_index = sorted_models[i % len(sorted_models)]
                    ideal_jobs[model_index] += 1 if diff > 0 else -1

                    # Ensure minimum of 2 jobs per model
                    if ideal_jobs[model_index] < 2:
                        ideal_jobs[model_index] = 2

            # Update target job counts
            model_jobs_target.update(ideal_jobs)

        balance_headers = [
            "Model",
            "Current Jobs",
            "Target Jobs",
            "Total Requests",
            "Request %",
            "Job %",
        ]

        # If no requests yet, show a simpler table
        if total_requests == 0:
            balance_headers = ["Model", "Current Jobs", "Target Jobs"]

            for model_index in sorted(model_jobs_target.keys()):
                model_name = model_names.get(model_index, f"Model-{model_index}")
                short_name = get_short_model_name(model_name)
                display_name = f"{model_index}:{short_name}"
                current = model_jobs_current.get(model_index, 0)
                target = model_jobs_target.get(model_index, 0)

                balance_table.append([display_name, current, target])
        else:
            for model_index in sorted(model_jobs_target.keys()):
                model_name = model_names.get(model_index, f"Model-{model_index}")
                short_name = get_short_model_name(model_name)
                display_name = f"{model_index}:{short_name}"
                current = model_jobs_current.get(model_index, 0)
                target = model_jobs_target.get(model_index, 0)
                requests = model_request_counts.get(model_index, 0)

                # Calculate percentages
                request_pct = (
                    (requests / total_requests * 100) if total_requests > 0 else 0
                )
                job_pct = (current / total_jobs * 100) if total_jobs > 0 else 0

                balance_table.append(
                    [
                        display_name,
                        current,
                        target,
                        requests,
                        f"{request_pct:.1f}%",
                        f"{job_pct:.1f}%",
                    ]
                )
    else:
        # Legacy mode
        balance_headers = ["Model", "Current Jobs", "Target Jobs"]

        for model_index, model_config in sorted(
            [(k, v) for k, v in config.items() if isinstance(k, int)]
        ):
            if "number_of_jobs" in model_config:
                model_name = model_names.get(model_index, model_config["model_name"])
                short_name = get_short_model_name(model_name)
                display_name = f"{model_index}:{short_name}"
                current = model_jobs_current.get(model_index, 0)
                target = model_config["number_of_jobs"]

                balance_table.append([display_name, current, target])

    print(colorize("\nModel Balance Status:", Colors.HEADER))
    print(tabulate(balance_table, headers=balance_headers, tablefmt="pretty"))


def print_status_header():
    """Print a status header with timestamp."""
    header = "=" * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(colorize("\n" + header, Colors.HEADER))
    print(
        colorize(
            f" vLLM System Status at {timestamp} ",
            Colors.HEADER + Colors.BOLD,
        ).center(100, "=")
    )
    print(colorize(header, Colors.HEADER))


def print_full_status(job_prefix="v"):
    """Print all status information."""
    print_status_header()
    print_model_balance_status()
    print_job_status(job_prefix)
    print_backend_status()
    print("")  # Add empty line after all tables


async def monitor_backends(job_prefix="v"):
    """Periodically update backend information."""
    while True:
        try:
            checked_models = await update_backend_info(job_prefix)
            if checked_models:
                print_full_status(job_prefix)
        except Exception as e:
            log_once(f"Error in backend monitoring: {e}", Colors.RED)
        await asyncio.sleep(MONITOR_INTERVAL)


async def monitor_jobs(test_mode=False, job_prefix="v"):
    """Monitor and resubmit jobs as needed."""
    username = os.environ.get("USER", subprocess.getoutput("whoami"))
    job_mapping = (
        {}
    )  # {(model_index, job_id): (job_name, port, model_name, model_name, gpu)}
    current_dir = os.getcwd()

    # Initialize job mapping
    for model_index, model_config in config.items():
        if isinstance(model_index, int):
            model_name = model_config["model_name"]
            gpu = model_config["gpu"]

            # If in legacy mode with per-model job counts
            if "number_of_jobs" in model_config:
                num_jobs = model_config["number_of_jobs"]
                for job_id in range(1, num_jobs + 1):
                    job_mapping[(model_index, job_id)] = (
                        None,
                        None,
                        model_name,
                        None,
                        gpu,
                    )

    # In dynamic balancing mode, we don't pre-populate job_mapping
    # We'll determine which jobs to submit based on the current balance

    last_job_check = 0

    while True:
        try:
            current_time = time.time()
            current_jobs = get_all_vllm_jobs(username, job_prefix)

            # Update job_mapping with current jobs
            for job_name in current_jobs:
                model_index, job_id, port = extract_job_info(job_name)

                if model_index is not None and job_id is not None:
                    if (model_index, job_id) in job_mapping:
                        model_name = job_mapping[(model_index, job_id)][2]
                        gpu = job_mapping[(model_index, job_id)][4]
                        job_mapping[(model_index, job_id)] = (
                            job_name,
                            port,
                            model_name,
                            model_name,
                            gpu,
                        )

            # Check for jobs that need to be submitted (only for legacy mode)
            if "total_number_of_jobs" not in config:
                for (model_index, job_id), (
                    job_name,
                    port,
                    model_name,
                    _,
                    gpu,
                ) in sorted(job_mapping.items()):
                    if job_name is None or not job_exists(job_name, username):
                        # Job doesn't exist, needs to be submitted
                        status_msg = (
                            "Resubmitting" if job_name else "Submitting new job"
                        )
                        print(
                            colorize(
                                f"Job {job_name or 'not found'} (Model {model_index}, Job {job_id}). {status_msg}...",
                                Colors.YELLOW,
                            )
                        )

                        success, new_job_name, new_port, new_model_name = submit_job(
                            model_index,
                            job_id,
                            model_name,
                            gpu,
                            current_dir,
                            username,
                            test_mode,
                            job_prefix,
                        )

                        if success:
                            job_mapping[(model_index, job_id)] = (
                                new_job_name,
                                new_port,
                                model_name,
                                new_model_name,
                                gpu,
                            )

            # Check if we need to submit new jobs based on dynamic balancing
            if "total_number_of_jobs" in config:
                # Count current jobs
                total_current_jobs = sum(model_jobs_current.values())

                # If we have fewer jobs than total_number_of_jobs, submit more
                jobs_to_submit = total_number_of_jobs - total_current_jobs

                if jobs_to_submit > 0:
                    print(
                        colorize(
                            f"Need to submit {jobs_to_submit} more jobs based on dynamic balancing",
                            Colors.CYAN,
                        )
                    )

                    for _ in range(jobs_to_submit):
                        # Choose which model to submit a job for
                        model_index = choose_model_for_new_job()

                        if model_index is not None and model_index in config:
                            # Double-check that we haven't exceeded target for this model
                            current = model_jobs_current.get(model_index, 0)
                            target = model_jobs_target.get(model_index, 0)

                            if current >= target:
                                print(
                                    colorize(
                                        f"Skipping job creation for model {model_index}: already at/above target ({current}/{target})",
                                        Colors.YELLOW,
                                    )
                                )
                                continue

                            model_name = config[model_index]["model_name"]
                            gpu = config[model_index]["gpu"]

                            # Get next available job ID
                            job_id = get_next_job_id(model_index, job_prefix)

                            print(
                                colorize(
                                    f"Dynamically submitting new job for model {model_index} (job ID: {job_id})",
                                    Colors.YELLOW,
                                )
                            )

                            success, job_name, port, _ = submit_job(
                                model_index,
                                job_id,
                                model_name,
                                gpu,
                                current_dir,
                                username,
                                test_mode,
                                job_prefix,
                            )

                            if success:
                                job_mapping[(model_index, job_id)] = (
                                    job_name,
                                    port,
                                    model_name,
                                    model_name,
                                    gpu,
                                )
                # Check if any model has more jobs than target and should be reduced
                # Note: This is informational only - we don't automatically kill jobs
                for model_index, target in model_jobs_target.items():
                    current = model_jobs_current.get(model_index, 0)
                    if current > target:
                        log_once(
                            f"Model {model_index} has more jobs ({current}) than target ({target}). Consider manual rebalancing.",
                            Colors.YELLOW,
                        )

            # Update time of last job check
            last_job_check = current_time
            await asyncio.sleep(MONITOR_INTERVAL)

        except Exception as e:
            log_once(f"Error in job monitoring: {e}", Colors.RED)
            await asyncio.sleep(MONITOR_INTERVAL)


# API Routes
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup and teardown for FastAPI."""
    # Don't start tasks here - main() will handle that
    yield


async def select_backend(model):
    """Select the best backend for a given model optimizing for reliability and performance."""
    global model_request_counts

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

    # Increment the request count for this model
    model_request_counts[target_model_index] = (
        model_request_counts.get(target_model_index, 0) + 1
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
        log_once(
            f"No healthy backends found for model '{model}', attempting recovery...",
            Colors.YELLOW,
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
            log_once(
                f"Using DEGRADED backend for model '{model}' as fallback",
                Colors.YELLOW,
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
            # Try checking each INITIALIZING backend - force check allowed for INITIALIZING
            for backend in initializing_backends:
                node, port = backend["node"], backend["port"]
                backend_key = f"{node}:{port}"

                # Try to check if it's ready now - force check
                is_healthy, _ = await check_backend_health(node, port, force_check=True)
                if is_healthy:
                    log_once(
                        f"Using newly healthy backend for model '{model}'",
                        Colors.GREEN,
                    )
                    backend_requests[backend_key] = (
                        backend_requests.get(backend_key, 0) + 1
                    )
                    backend_in_flight[backend_key] = (
                        backend_in_flight.get(backend_key, 0) + 1
                    )
                    return backend

        # If no DEGRADED or ready INITIALIZING backends, try health checking UNHEALTHY ones
        # But respect the check intervals - only check if due
        now = time.time()
        for backend in backends[target_model_index]:
            node, port = backend["node"], backend["port"]
            backend_key = f"{node}:{port}"
            current_state = backend_states.get(backend_key)

            # Skip permanently DEAD backends
            if current_state == BackendState.DEAD:
                continue

            # Only check if the backend is due for checking
            if (
                backend_key in backend_check_times
                and now < backend_check_times[backend_key]
            ):
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
        # This is a payload failure, not a health check failure
        # Call the payload-specific handler which only allows HEALTHY → DEGRADED transition
        handle_payload_failure(backend_key, str(e))

        log_once(f"Backend request failed: {e}", Colors.RED)
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
                                    log_once(
                                        f"Warning: First token from {backend_key} took {time_to_first_token:.2f}s (threshold: {REQUEST_STREAM_TIMEOUT}s)",
                                        Colors.YELLOW,
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
                    # Handle timeout for first token - this is a payload failure
                    if (
                        not first_token_received
                        and (time.time() - start_time) > REQUEST_STREAM_TIMEOUT
                    ):
                        log_once(
                            f"Streaming timeout: No response from {backend_key} after {REQUEST_STREAM_TIMEOUT}s",
                            Colors.RED,
                        )

                        # This is a payload failure
                        handle_payload_failure(backend_key, "Streaming timeout")

                        # Return error message to client
                        yield json.dumps(
                            {
                                "error": f"Streaming timeout: No response after {REQUEST_STREAM_TIMEOUT}s"
                            }
                        ).encode()
                except Exception as e:
                    # Handle streaming exception - this is a payload failure
                    handle_payload_failure(backend_key, str(e))

                    log_once(f"Backend streaming request failed: {e}", Colors.RED)
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


def create_api_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(title="vLLM API Proxy", lifespan=lifespan)

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
                        # Log retry attempt (deduplicated)
                        log_once(
                            f"Retry {retry_count}/{max_retries} for request to model '{model}' after error: {str(e)}",
                            Colors.YELLOW,
                        )
                        # Short delay before retry
                        await asyncio.sleep(0.5)
                    else:
                        # Max retries exceeded
                        break

            # If we get here, all retries failed
            error_message = f"All {max_retries} retries failed for model '{model}': {str(last_error)}"
            log_once(error_message, Colors.RED)
            return JSONResponse(content={"error": error_message}, status_code=503)

        except HTTPException as e:
            return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
        except Exception as e:
            log_once(f"Error processing request: {e}", Colors.RED)
            return JSONResponse(
                content={"error": f"Error processing request: {str(e)}"},
                status_code=500,
            )

    return app


async def run_api(port):
    """Run the API server."""
    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level="error", access_log=False
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main_async(args):
    """Main async function that orchestrates all components."""
    global app, MONITOR_INTERVAL, config

    # Update monitor interval
    MONITOR_INTERVAL = args.monitor_interval

    # Load configuration
    load_config(args.config)

    # Create API app
    app = create_api_app()

    # Prepare tasks based on command line arguments
    tasks = []

    if not args.job_only:
        # Include API and backend monitoring tasks
        tasks.append(run_api(args.port))
        tasks.append(monitor_backends(args.job_prefix))

    if not args.api_only:
        # Include job monitoring task
        tasks.append(monitor_jobs(args.test_mode, args.job_prefix))

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


def main():
    """Main entry point."""
    args = parse_arguments()
    username = os.environ.get("USER", subprocess.getoutput("whoami"))

    # Handle cancel_all flag first
    if args.cancel_all:
        cancel_all_vllm_jobs(username, args.job_prefix)
        return

    # Print banner
    print(colorize("=" * 100, Colors.HEADER))
    print(
        colorize(
            " vLLM Management System with Dynamic Load Balancing ",
            Colors.HEADER + Colors.BOLD,
        ).center(100, "=")
    )
    print(colorize("=" * 100, Colors.HEADER))

    # Load configuration early to print summary
    load_config(args.config)

    # Print operation mode
    if args.api_only:
        print(colorize("Running in API-only mode", Colors.CYAN))
    elif args.job_only:
        print(colorize("Running in job management-only mode", Colors.CYAN))
    else:
        print(colorize("Running in combined mode (API + job management)", Colors.GREEN))

    # Print configuration summary
    print(colorize(f"Configuration file: {args.config}", Colors.CYAN))
    print(colorize(f"API port: {args.port}", Colors.CYAN))
    print(colorize(f"Job prefix: '{args.job_prefix}'", Colors.CYAN))
    print(colorize(f"Monitor interval: {args.monitor_interval} seconds", Colors.CYAN))

    # Print model configuration
    model_table = []
    model_headers = ["Model Index", "Model Name", "GPU"]

    for model_index, model_config in sorted(
        [(k, v) for k, v in config.items() if isinstance(k, int)]
    ):
        model_table.append(
            [model_index, model_config["model_name"], model_config["gpu"]]
        )

    print(colorize("\nConfigured Models:", Colors.HEADER))
    print(tabulate(model_table, headers=model_headers, tablefmt="pretty"))

    # Print job allocation strategy
    if "total_number_of_jobs" in config:
        print(
            colorize(
                f"\nDynamic job balancing enabled with {config['total_number_of_jobs']} total jobs",
                Colors.GREEN,
            )
        )
        print(
            colorize("Jobs will be allocated based on request patterns", Colors.GREEN)
        )
    else:
        print(
            colorize("\nLegacy mode: using fixed job counts per model", Colors.YELLOW)
        )

    print(colorize("=" * 100, Colors.HEADER))

    if args.test_mode:
        print(colorize("\n=== RUNNING IN TEST MODE ===", Colors.RED + Colors.BOLD))
        print(
            colorize(
                "Jobs will terminate at staggered intervals for testing", Colors.RED
            )
        )
        print(colorize("=" * 100, Colors.HEADER))

    # Run the async main function
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print(colorize("\nShutting down...", Colors.YELLOW))


if __name__ == "__main__":
    main()
