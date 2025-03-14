#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import tempfile
import time
import random
import yaml
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
import re


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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=colorize(
            "Script to launch vLLM server jobs using Slurm", Colors.HEADER
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=30,
        help="Interval in seconds between job status checks",
    )
    parser.add_argument(
        "--cancel-all",
        action="store_true",
        help="Cancel all vLLM jobs and exit",
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(colorize(f"Error loading configuration: {e}", Colors.RED))
        sys.exit(1)


def get_model_short_name(model_name):
    """Extract a short identifier from the model name."""
    # Extract the model name after the last slash or use the whole name
    short_name = model_name.split("/")[-1]

    # Extract the base model name (before version numbers)
    match = re.search(r"^([A-Za-z]+)", short_name)
    if match:
        return match.group(1).lower()[:3]  # Take first 3 chars of the model name

    # Fallback: take first 3 chars
    return short_name[:3].lower()


def get_available_port(used_ports):
    """Get a port number that is not in the used_ports list."""
    # Only consider ports in the 9000-9999 range
    # Exclude common ports like 9000, 9090, 9999
    excluded_ports = {9000, 9090, 9999}

    # Create a list of potential ports
    potential_ports = [p for p in range(9000, 10000) if p not in excluded_ports]

    # Shuffle the list to reduce collision probability
    random.shuffle(potential_ports)

    # Return the first port that is not in used_ports
    for port in potential_ports:
        if port not in used_ports:
            return port

    # If all ports are used (unlikely), return a random port
    return random.randint(9000, 9999)


def get_used_ports(username):
    """Get a list of ports used by existing vLLM jobs."""
    used_ports = []

    # Get all jobs for the user
    cmd = ["squeue", "-u", username, "-h", "-o", "%j"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract port numbers from job names
    for job_name in result.stdout.strip().split("\n"):
        if job_name and job_name.startswith("v"):
            try:
                # Extract the port from the job name (last 3 digits)
                port_suffix = job_name[-3:]
                port = int("9" + port_suffix)  # Add "9" prefix to make it 9XXX
                used_ports.append(port)
            except (ValueError, IndexError):
                pass

    return used_ports


def ensure_log_dir():
    """Ensure the log directory exists."""
    log_dir = Path("./vllm_logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def submit_job(model_index, job_id, model_name, gpu, current_dir, username):
    """Submit a vLLM job to Slurm."""
    # Define common parameters for all jobs
    qos = "scavenger"
    time_limit = "01:00:00"
    cpu = "4"  # Reasonable for 1 GPU
    mem = "16G"  # Reasonable for 1 GPU
    partition = "--partition=scavenger"
    account = "--account=scavenger"

    # Get a port that is not used by existing jobs
    used_ports = get_used_ports(username)
    port = get_available_port(used_ports)

    # Extract the last 3 digits of the port
    port_suffix = str(port)[-3:]

    # Format model index and job ID with leading zeros
    formatted_model_index = f"{model_index:02d}"
    formatted_job_id = f"{job_id:02d}"

    # Create job name with model index, job ID, and port suffix
    job_name = f"v{formatted_model_index}{formatted_job_id}{port_suffix}"

    # Ensure log directory exists
    log_dir = ensure_log_dir()

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a temporary job script
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file_path = temp_file.name

        # Write the Slurm job script
        job_script = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --qos={qos}
#SBATCH {partition}
#SBATCH {account}
#SBATCH --time={time_limit}
#SBATCH --gres={gpu}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --mem={mem}
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
echo "========================================================"

vllm serve {model_name} --port {port}
"""
        temp_file.write(job_script)

    # Submit the job
    print(
        colorize(
            f"Submitting vLLM job {job_name} for model {model_name}...",
            Colors.CYAN,
        )
    )
    result = subprocess.run(["sbatch", temp_file_path], capture_output=True, text=True)

    # Clean up the temporary file
    os.unlink(temp_file_path)

    if result.returncode == 0:
        print(colorize(f"Job submitted: {result.stdout.strip()}", Colors.GREEN))
        return True, job_name, port, model_name
    else:
        print(colorize(f"Error submitting job: {result.stderr}", Colors.RED))
        return False, job_name, port, model_name


def extract_job_info(job_name):
    """Extract model index, job ID and port from job name."""
    try:
        # Format: v0102XXX where 01 is model index, 02 is job ID, and XXX is port suffix
        model_index = int(job_name[1:3])
        job_id = int(job_name[3:5])
        port_suffix = job_name[5:]
        port = int("9" + port_suffix)  # Add "9" prefix to make it 9XXX
        return model_index, job_id, port
    except (ValueError, IndexError):
        return None, None, None


def get_job_id_by_name(job_name, username):
    """Get the Slurm job ID for a job with the given name."""
    cmd = ["squeue", "-u", username, "-n", job_name, "-h", "-o", "%i"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Return the job ID if found, otherwise None
    job_id = result.stdout.strip()
    return job_id if job_id else None


def get_job_status(job_name, username):
    """Get detailed status of a job."""
    # Get full job information
    cmd = ["squeue", "-u", username, "-n", job_name, "-h", "-o", "%i|%j|%T|%N|%M|%L"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if not result.stdout.strip():
        return None

    # Parse the output
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
    # Method 1: Use squeue with name filter
    cmd1 = ["squeue", "-u", username, "-n", job_name, "-h"]
    result1 = subprocess.run(cmd1, capture_output=True, text=True)

    # Method 2: Use squeue with name filter and state filter for running and pending
    cmd2 = ["squeue", "-u", username, "-n", job_name, "-t", "RUNNING,PENDING", "-h"]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)

    # Method 3: Get the actual job ID if it exists
    slurm_job_id = get_job_id_by_name(job_name, username)

    # If any method finds the job, consider it exists
    return bool(result1.stdout.strip() or result2.stdout.strip() or slurm_job_id)


def get_all_vllm_jobs(username):
    """Get all vLLM jobs for the user."""
    cmd = ["squeue", "-u", username, "-h", "-o", "%j"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    vllm_jobs = []
    for job_name in result.stdout.strip().split("\n"):
        if job_name and job_name.startswith("v"):
            vllm_jobs.append(job_name)

    return vllm_jobs


def monitor_jobs(config, monitor_interval, current_dir):
    """Monitor and resubmit jobs as needed."""
    username = os.environ.get("USER", subprocess.getoutput("whoami"))

    # Dictionary to keep track of job names and ports
    job_mapping = (
        {}
    )  # {(model_index, job_id): (job_name, port, model_name, model_name, gpu)}

    # Initialize job_mapping with configuration
    for model_index, model_config in config.items():
        model_name = model_config["model_name"]
        num_jobs = model_config["number_of_jobs"]
        gpu = model_config["gpu"]

        for job_id in range(1, num_jobs + 1):
            job_mapping[(model_index, job_id)] = (None, None, model_name, None, gpu)

    print(
        colorize(
            f"\nStarting job monitoring. Will check every {monitor_interval} seconds...",
            Colors.CYAN,
        )
    )
    print(colorize("Press Ctrl+C to stop monitoring and exit.\n", Colors.YELLOW))

    # Track when we last printed the status header
    last_header_time = 0
    header_interval = 300  # Print header every 5 minutes

    try:
        while True:
            current_time = time.time()

            # Get all current vLLM jobs
            current_jobs = get_all_vllm_jobs(username)

            # Update job_mapping with current jobs
            for job_name in current_jobs:
                model_index, job_id, port = extract_job_info(job_name)
                if model_index is not None and job_id is not None:
                    if (model_index, job_id) in job_mapping:
                        # Update existing job
                        model_name = job_mapping[(model_index, job_id)][2]
                        gpu = job_mapping[(model_index, job_id)][4]
                        job_mapping[(model_index, job_id)] = (
                            job_name,
                            port,
                            model_name,
                            model_name,
                            gpu,
                        )

            # Print status header if needed
            if current_time - last_header_time > header_interval:
                print(colorize("\n" + "=" * 100, Colors.HEADER))
                print(
                    colorize(
                        f" vLLM Job Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ",
                        Colors.HEADER + Colors.BOLD,
                    ).center(100, "=")
                )
                print(colorize("=" * 100, Colors.HEADER))
                last_header_time = current_time

            # Prepare status table
            status_table = []
            status_headers = ["Model", "Job", "Status", "Node", "Runtime", "Port"]

            # Check each job and resubmit if needed
            for (model_index, job_id), (
                job_name,
                port,
                model_name,
                _,
                gpu,
            ) in sorted(job_mapping.items()):
                if job_name is not None and job_exists(job_name, username):
                    # Job exists, get status
                    job_status = get_job_status(job_name, username)
                    if job_status:
                        state = job_status["state"]
                        state_color = (
                            Colors.GREEN if state == "RUNNING" else Colors.YELLOW
                        )

                        # Add to status table
                        status_table.append(
                            [
                                f"{model_index}:{model_name}",
                                job_id,
                                colorize(state, state_color),
                                job_status["node"],
                                job_status["time"],
                                port,
                            ]
                        )
                    else:
                        # Job exists but couldn't get status
                        status_table.append(
                            [
                                f"{model_index}:{model_name}",
                                job_id,
                                colorize("UNKNOWN", Colors.YELLOW),
                                "N/A",
                                "N/A",
                                port,
                            ]
                        )
                else:
                    # Job doesn't exist or needs to be submitted/resubmitted
                    if job_name is not None:
                        print(
                            colorize(
                                f"Job {job_name} (Model {model_index}, Job {job_id}) is not running. Resubmitting...",
                                Colors.YELLOW,
                            )
                        )
                    else:
                        print(
                            colorize(
                                f"No job found for Model {model_index}, Job {job_id}. Submitting new job...",
                                Colors.YELLOW,
                            )
                        )

                    success, new_job_name, new_port, new_model_name = submit_job(
                        model_index, job_id, model_name, gpu, current_dir, username
                    )

                    if success:
                        job_mapping[(model_index, job_id)] = (
                            new_job_name,
                            new_port,
                            model_name,
                            new_model_name,
                            gpu,
                        )

                        # Add to status table
                        status_table.append(
                            [
                                f"{model_index}:{model_name}",
                                job_id,
                                colorize("SUBMITTED", Colors.BLUE),
                                "pending",
                                "0:00",
                                new_port,
                            ]
                        )

            # Print status table if not empty
            if status_table:
                print(tabulate(status_table, headers=status_headers, tablefmt="pretty"))
                print("")

            # Wait before checking again
            time.sleep(monitor_interval)
    except KeyboardInterrupt:
        print(colorize("\nMonitoring stopped by user.", Colors.YELLOW))


def cancel_all_vllm_jobs(username):
    """Cancel all vLLM jobs for the user."""
    print(colorize("\nCancelling all vLLM jobs...", Colors.YELLOW))

    # Get all vLLM jobs
    current_jobs = get_all_vllm_jobs(username)

    if not current_jobs:
        print(colorize("No vLLM jobs found to cancel.", Colors.CYAN))
        return

    # Print jobs to be cancelled
    print(colorize(f"Found {len(current_jobs)} vLLM jobs to cancel:", Colors.CYAN))
    for job_name in current_jobs:
        print(colorize(f"  - {job_name}", Colors.CYAN))

    # Cancel all jobs with names starting with v
    cmd = ["scancel", "-n", "v*", "-u", username]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(colorize("All vLLM jobs have been cancelled successfully.", Colors.GREEN))
    else:
        print(colorize(f"Error cancelling jobs: {result.stderr}", Colors.RED))

    # Verify cancellation
    time.sleep(2)  # Wait a bit for the cancellation to take effect
    remaining_jobs = get_all_vllm_jobs(username)

    if remaining_jobs:
        print(
            colorize(
                f"Warning: {len(remaining_jobs)} vLLM jobs still remain:", Colors.YELLOW
            )
        )
        for job_name in remaining_jobs:
            print(colorize(f"  - {job_name}", Colors.YELLOW))

        # Try cancelling by job ID as a fallback
        print(
            colorize("Attempting to cancel remaining jobs by job ID...", Colors.YELLOW)
        )
        for job_name in remaining_jobs:
            job_id = get_job_id_by_name(job_name, username)
            if job_id:
                cancel_cmd = ["scancel", job_id]
                subprocess.run(cancel_cmd)
                print(
                    colorize(f"Cancelled job {job_name} (ID: {job_id})", Colors.GREEN)
                )
    else:
        print(colorize("All jobs have been successfully cancelled.", Colors.GREEN))


def main():
    args = parse_arguments()

    # Get username
    username = os.environ.get("USER", subprocess.getoutput("whoami"))

    # Check if we should cancel all jobs and exit
    if args.cancel_all:
        cancel_all_vllm_jobs(username)
        return

    # Load configuration
    config = load_config(args.config)

    # Get current directory
    current_dir = os.getcwd()

    # Ensure log directory exists
    ensure_log_dir()

    # Print configuration
    print(colorize("=" * 100, Colors.HEADER))
    print(
        colorize(
            " vLLM Multi-Model Job Configuration ", Colors.HEADER + Colors.BOLD
        ).center(100, "=")
    )
    print(colorize("=" * 100, Colors.HEADER))

    # Prepare configuration table
    config_table = []
    config_headers = ["Model Index", "Model Name", "Jobs", "GPU"]

    total_jobs = 0
    for model_index, model_config in sorted(config.items()):
        model_name = model_config["model_name"]
        num_jobs = model_config["number_of_jobs"]
        gpu = model_config["gpu"]

        config_table.append([model_index, model_name, num_jobs, gpu])

        total_jobs += num_jobs

    # Print configuration table
    print(tabulate(config_table, headers=config_headers, tablefmt="pretty"))
    print(colorize(f"\nTotal jobs to be submitted: {total_jobs}", Colors.CYAN))
    print(colorize(f"Monitor interval: {args.monitor_interval} seconds", Colors.CYAN))
    print(colorize(f"Log directory: ./vllm_logs", Colors.CYAN))
    print(colorize("=" * 100, Colors.HEADER))

    # Launch jobs for each model
    job_mapping = {}  # {(model_index, job_id): (job_name, port, model_name)}

    print(colorize("\nSubmitting jobs...", Colors.YELLOW))

    for model_index, model_config in sorted(config.items()):
        model_name = model_config["model_name"]
        num_jobs = model_config["number_of_jobs"]
        gpu = model_config["gpu"]

        print(
            colorize(
                f"\nSubmitting {num_jobs} jobs for model {model_index}: {model_name}",
                Colors.CYAN,
            )
        )

        for job_id in range(1, num_jobs + 1):
            success, job_name, port, model_name_full = submit_job(
                model_index, job_id, model_name, gpu, current_dir, username
            )

            if success:
                job_mapping[(model_index, job_id)] = (job_name, port, model_name)

            # Wait a bit between submissions
            time.sleep(1)

    print(colorize(f"\nAll {total_jobs} vLLM jobs submitted.", Colors.GREEN))

    # Print job information
    print(colorize("\nJob Information:", Colors.HEADER))

    # Prepare job information table
    job_info_table = []
    job_info_headers = ["Model", "Job ID", "Job Name", "Port"]

    for (model_index, job_id), (job_name, port, model_name) in sorted(
        job_mapping.items()
    ):
        job_info_table.append([f"{model_index}:{model_name}", job_id, job_name, port])

    # Print job information table
    print(tabulate(job_info_table, headers=job_info_headers, tablefmt="pretty"))

    print(colorize("\nStarting job monitoring...", Colors.YELLOW))

    # Start monitoring in the foreground
    monitor_jobs(config, args.monitor_interval, current_dir)


if __name__ == "__main__":
    main()
