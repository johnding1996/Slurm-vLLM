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

# Test mode configuration
TEST_BASE_TIME = 120  # Initial base time (seconds)
TEST_INTERVAL = 30  # Time interval between job failures (seconds)

# Track job submission counts for test mode
test_job_submission_counts = {}  # {(model_index, job_id): count}


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
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with simulated job failures at regular intervals",
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(colorize(f"Error loading configuration: {e}", Colors.RED))
        sys.exit(1)


def get_available_port(used_ports):
    """Get a port number that is not in the used_ports list."""
    excluded_ports = {9000, 9090, 9999}
    potential_ports = [p for p in range(9000, 10000) if p not in excluded_ports]
    random.shuffle(potential_ports)

    for port in potential_ports:
        if port not in used_ports:
            return port

    return random.randint(9000, 9999)


def get_used_ports(username):
    """Get a list of ports used by existing vLLM jobs."""
    used_ports = []
    cmd = ["squeue", "-u", username, "-h", "-o", "%j"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for job_name in result.stdout.strip().split("\n"):
        if job_name and job_name.startswith("v"):
            try:
                port_suffix = job_name[-3:]
                used_ports.append(int("9" + port_suffix))
            except (ValueError, IndexError):
                pass

    return used_ports


def ensure_log_dir():
    """Ensure the log directory exists."""
    log_dir = Path("./vllm_logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def submit_job(
    model_index,
    job_id,
    model_name,
    gpu,
    current_dir,
    username,
    test_mode=False,
    num_jobs_in_model=None,
):
    """Submit a vLLM job to Slurm."""
    # Slurm job parameters
    slurm_params = {
        "qos": "scavenger",
        "time_limit": "01:00:00",
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
        num_jobs = num_jobs_in_model if num_jobs_in_model is not None else 6

        if submission_count == 0:
            # For first submission, stagger time limits based on job_id
            # Job 1: BASE_TIME, Job 2: BASE_TIME + INTERVAL, etc.
            time_limit_seconds = TEST_BASE_TIME + ((job_id - 1) * TEST_INTERVAL)
        else:
            # For resubmissions, always use a fixed time interval = 30s * number_of_jobs
            time_limit_seconds = TEST_INTERVAL * num_jobs

        # Format time as HH:MM:SS
        minutes, seconds = divmod(time_limit_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        slurm_params["time_limit"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    used_ports = get_used_ports(username)
    port = get_available_port(used_ports)
    port_suffix = str(port)[-3:]

    formatted_model_index = f"{model_index:02d}"
    formatted_job_id = f"{job_id:02d}"
    job_name = f"v{formatted_model_index}{formatted_job_id}{port_suffix}"

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
                    f"  [TEST MODE] Time limit: {slurm_params['time_limit']} (Submission #{submission_count+1})",
                    Colors.YELLOW,
                )
            )
        return True, job_name, port, model_name
    else:
        print(colorize(f"Error submitting job: {result.stderr}", Colors.RED))
        return False, job_name, port, model_name


def extract_job_info(job_name):
    """Extract model index, job ID and port from job name."""
    try:
        # Basic format is v{model_index:02d}{job_id:02d}{port_suffix}
        # Where port_suffix is the last 3 digits of the port number
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


def get_all_vllm_jobs(username):
    """Get all vLLM jobs for the user."""
    cmd = ["squeue", "-u", username, "-h", "-o", "%j"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return [
        job_name
        for job_name in result.stdout.strip().split("\n")
        if job_name and job_name.startswith("v")
    ]


def print_status_header():
    """Print a status header for the job status table."""
    header = "=" * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(colorize("\n" + header, Colors.HEADER))
    print(
        colorize(
            f" vLLM Job Status at {timestamp} ",
            Colors.HEADER + Colors.BOLD,
        ).center(100, "=")
    )
    print(colorize(header, Colors.HEADER))


def get_short_model_name(model_name):
    """Create a short, readable version of model name.
    For each part (separated by / or -):
    - Keep entire part if ≤ 4 chars
    - Use first 4 chars if longer
    - Preserve all / and - characters
    """
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


def monitor_jobs(config, monitor_interval, current_dir, test_mode=False):
    """Monitor and resubmit jobs as needed."""
    username = os.environ.get("USER", subprocess.getoutput("whoami"))
    job_mapping = (
        {}
    )  # {(model_index, job_id): (job_name, port, model_name, model_name, gpu)}

    # Initialize job mapping
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
    if test_mode:
        print(
            colorize(
                "TEST MODE ACTIVE: Jobs will timeout at staggered intervals", Colors.RED
            )
        )
    print(colorize("Press Ctrl+C to stop monitoring and exit.\n", Colors.YELLOW))

    last_header_time = 0
    header_interval = 300  # Print header every 5 minutes

    try:
        while True:
            current_time = time.time()
            current_jobs = get_all_vllm_jobs(username)

            # Update job_mapping with current jobs
            for job_name in current_jobs:
                base_job_name = job_name
                model_index, job_id, port = extract_job_info(base_job_name)

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

            # Print status header if needed
            if current_time - last_header_time > header_interval:
                print_status_header()
                last_header_time = current_time

            status_table = []
            status_headers = [
                "Model",
                "Job",
                "Status",
                "Node",
                "Runtime",
                "Port",
                "Resubmits",
            ]

            # Check each job and resubmit if needed
            for (model_index, job_id), (job_name, port, model_name, _, gpu) in sorted(
                job_mapping.items()
            ):
                if job_name is not None and job_exists(job_name, username):
                    job_status = get_job_status(job_name, username)

                    # Get resubmit count - use 0 if not in test mode or key doesn't exist
                    key = (model_index, job_id)
                    resubmit_count = test_job_submission_counts.get(key, 0)

                    if job_status:
                        state = job_status["state"]
                        state_color = (
                            Colors.GREEN if state == "RUNNING" else Colors.YELLOW
                        )

                        # Create short model name for display
                        short_model = get_short_model_name(model_name)

                        status_row = [
                            f"{model_index}:{short_model}",
                            job_id,
                            colorize(state, state_color),
                            job_status["node"],
                            job_status["time"],
                            port,
                            resubmit_count,
                        ]

                        status_table.append(status_row)
                    else:
                        # Create short model name for display
                        short_model = get_short_model_name(model_name)

                        status_row = [
                            f"{model_index}:{short_model}",
                            job_id,
                            colorize("UNKNOWN", Colors.YELLOW),
                            "N/A",
                            "N/A",
                            port,
                            resubmit_count,
                        ]

                        status_table.append(status_row)
                else:
                    # Job doesn't exist, needs to be submitted
                    status_msg = "Resubmitting" if job_name else "Submitting new job"
                    print(
                        colorize(
                            f"Job {job_name or 'not found'} (Model {model_index}, Job {job_id}). {status_msg}...",
                            Colors.YELLOW,
                        )
                    )

                    # Get the number of jobs for this model from config
                    model_config = config.get(model_index)
                    num_jobs_in_model = (
                        model_config["number_of_jobs"] if model_config else None
                    )

                    success, new_job_name, new_port, new_model_name = submit_job(
                        model_index,
                        job_id,
                        model_name,
                        gpu,
                        current_dir,
                        username,
                        test_mode,
                        num_jobs_in_model=num_jobs_in_model,
                    )

                    if success:
                        job_mapping[(model_index, job_id)] = (
                            new_job_name,
                            new_port,
                            model_name,
                            new_model_name,
                            gpu,
                        )

                        # Get resubmit count
                        key = (model_index, job_id)
                        resubmit_count = test_job_submission_counts.get(key, 0)

                        # Create short model name for display
                        short_model = get_short_model_name(model_name)

                        status_row = [
                            f"{model_index}:{short_model}",
                            job_id,
                            colorize("SUBMITTED", Colors.BLUE),
                            "pending",
                            "0:00",
                            new_port,
                            resubmit_count,
                        ]

                        status_table.append(status_row)

            if status_table:
                print(tabulate(status_table, headers=status_headers, tablefmt="pretty"))
                print("")

            time.sleep(monitor_interval)
    except KeyboardInterrupt:
        print(colorize("\nMonitoring stopped by user.", Colors.YELLOW))


def cancel_all_vllm_jobs(username):
    """Cancel all vLLM jobs for the user."""
    print(colorize("\nCancelling all vLLM jobs...", Colors.YELLOW))

    current_jobs = get_all_vllm_jobs(username)

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


def display_test_job_schedule(config):
    """Display the timeout schedule for test mode jobs."""
    print(colorize("\nTest Mode Job Timeout Schedule:", Colors.YELLOW))
    print(colorize("-----------------------------", Colors.YELLOW))

    # Calculate fixed resubmission timeout
    max_jobs = max(model_config["number_of_jobs"] for model_config in config.values())
    fixed_resubmission_timeout = TEST_INTERVAL * max_jobs

    headers = ["Job ID", "Initial Timeout", "Resubmission Timeout"]
    schedule = []

    for job_id in range(1, max_jobs + 1):
        # Initial timeout for this job
        initial_timeout = TEST_BASE_TIME + ((job_id - 1) * TEST_INTERVAL)

        schedule.append(
            [
                job_id,
                f"{initial_timeout}s (~{int(initial_timeout/60)}m {initial_timeout%60}s)",
                f"{fixed_resubmission_timeout}s (~{int(fixed_resubmission_timeout/60)}m {fixed_resubmission_timeout%60}s)",
            ]
        )

    print(tabulate(schedule, headers=headers, tablefmt="grid"))

    # Explanation of the timeout pattern
    print(colorize("\nTimeout Pattern Explanation:", Colors.CYAN))
    print("1. Initial jobs timeout at staggered intervals")
    print(
        f"2. ALL resubmitted jobs have a fixed timeout of {fixed_resubmission_timeout}s"
    )
    print(
        f"3. After initial timeouts, jobs will be resubmitted and timeout every {fixed_resubmission_timeout}s"
    )
    print("")


def main():
    args = parse_arguments()
    username = os.environ.get("USER", subprocess.getoutput("whoami"))

    if args.cancel_all:
        cancel_all_vllm_jobs(username)
        return

    config = load_config(args.config)
    current_dir = os.getcwd()
    ensure_log_dir()

    # Set monitor interval for test mode
    monitor_interval = args.monitor_interval

    # Initialize test mode if enabled
    if args.test_mode:
        # Calculate fixed resubmission timeout based on the highest number of jobs in any model
        max_jobs = max(
            model_config["number_of_jobs"] for model_config in config.values()
        )
        fixed_resubmission_timeout = TEST_INTERVAL * max_jobs

        # Use a shorter monitor interval in test mode for more responsive monitoring
        monitor_interval = 5

        print(colorize("\n=== RUNNING IN TEST MODE ===", Colors.RED + Colors.BOLD))
        print(
            colorize(
                f"Jobs will terminate at staggered intervals for testing",
                Colors.RED,
            )
        )
        print(
            colorize(
                f"First batch: BASE({TEST_BASE_TIME}s) + job_id*{TEST_INTERVAL}s",
                Colors.RED,
            )
        )
        print(
            colorize(
                f"All resubmissions: Fixed timeout of {TEST_INTERVAL}s * num_jobs for each model",
                Colors.RED,
            )
        )
        print(
            colorize(
                f"Using shorter monitor interval: {monitor_interval}s (default: {args.monitor_interval}s)",
                Colors.RED,
            )
        )
        print(colorize("================================\n", Colors.RED + Colors.BOLD))

        # Display test job schedule
        display_test_job_schedule(config)

    # Print configuration header
    print(colorize("=" * 100, Colors.HEADER))
    print(
        colorize(
            " vLLM Multi-Model Job Configuration ", Colors.HEADER + Colors.BOLD
        ).center(100, "=")
    )
    print(colorize("=" * 100, Colors.HEADER))

    # Build configuration table
    config_table = []
    config_headers = ["Model Index", "Model Name", "Short Name", "Jobs", "GPU"]
    total_jobs = 0

    for model_index, model_config in sorted(config.items()):
        model_name = model_config["model_name"]
        short_name = get_short_model_name(model_name)
        num_jobs = model_config["number_of_jobs"]
        gpu = model_config["gpu"]

        config_table.append([model_index, model_name, short_name, num_jobs, gpu])
        total_jobs += num_jobs

    print(tabulate(config_table, headers=config_headers, tablefmt="pretty"))
    print(colorize(f"\nTotal jobs to be submitted: {total_jobs}", Colors.CYAN))
    print(colorize(f"Monitor interval: {monitor_interval} seconds", Colors.CYAN))
    print(colorize(f"Log directory: ./vllm_logs", Colors.CYAN))
    print(colorize("=" * 100, Colors.HEADER))

    # Submit jobs
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
                model_index,
                job_id,
                model_name,
                gpu,
                current_dir,
                username,
                args.test_mode,
                num_jobs_in_model=num_jobs,
            )

            if success:
                job_mapping[(model_index, job_id)] = (job_name, port, model_name)

            time.sleep(1)

    print(colorize(f"\nAll {total_jobs} vLLM jobs submitted.", Colors.GREEN))

    # Print job information
    print(colorize("\nJob Information:", Colors.HEADER))
    job_info_table = []
    job_info_headers = ["Model", "Job ID", "Job Name", "Port"]

    for (model_index, job_id), (job_name, port, model_name) in sorted(
        job_mapping.items()
    ):
        job_info_table.append([f"{model_index}:{model_name}", job_id, job_name, port])

    print(tabulate(job_info_table, headers=job_info_headers, tablefmt="pretty"))
    print(colorize("\nStarting job monitoring...", Colors.YELLOW))

    # Start job monitoring
    monitor_jobs(config, monitor_interval, current_dir, args.test_mode)


if __name__ == "__main__":
    main()
