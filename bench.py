#!/usr/bin/env python3
# Continuous benchmarking tool for vLLM API Proxy
# Tests all models in config.yaml with various question types
# Shows real-time metrics and generates a final report when interrupted

import asyncio
import random
import signal
import time
import yaml
import sys
import os
from collections import defaultdict
from datetime import datetime, timedelta
from openai import AsyncOpenAI
from tabulate import tabulate
import argparse
import math


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


# Constants
BASE_URL = None  # Will be set based on API port argument
CONFIG_PATH = "config.yaml"
MAX_CONCURRENCY = 400  # Maximum concurrent requests
UPDATE_INTERVAL = 2  # Seconds between metric updates
RESPONSE_SAMPLE_SIZE = 2  # Number of example responses to keep per model/question type
MAX_TOKENS = 1000  # Increased max tokens to ensure complete answers
IMBALANCE_RATIO = 0  # Default imbalance ratio (0 = even distribution)


# Question types with examples
QUESTION_TYPES = {
    "easy_math": [
        "If John has 5 apples and Mary gives him 3 more, how many apples does John have in total?",
        "A train travels at 60 miles per hour. How far will it travel in 2.5 hours?",
        "If a shirt costs $25 and is on sale for 20% off, what is the sale price?",
        "Jane scored 80, 85, and 90 on her first three tests. What score does she need on her fourth test to have an average of 85?",
        "A recipe requires 2.5 cups of flour to make 24 cookies. How much flour is needed to make 60 cookies?",
    ],
    "hard_math": [
        "Solve for x: 3x^2 - 6x - 24 = 0",
        "A sphere has a volume of 288π cubic centimeters. What is its radius in centimeters?",
        "In a geometric sequence, the first term is 5 and the common ratio is 3. What is the sum of the first 8 terms?",
        "A car travels uphill at 40 km/h and downhill at 60 km/h. If the total journey is 120 km and takes 2 hours 24 minutes, how long is the uphill portion?",
        "Find all values of x where f(x) = sin(x) + cos(x) has a maximum value, and determine this maximum value.",
    ],
    "coding": [
        "Write a Python function to check if a string is a palindrome.",
        "Implement a function that finds the longest common subsequence of two strings.",
        "Create an algorithm to determine if a binary tree is balanced.",
        "Write a function to find all permutations of a given string without using any library functions.",
        "Implement a solution to the 'N-Queens' problem for an arbitrary N.",
    ],
    "logic": [
        "Three people - Alice, Bob, and Charlie - each tell either only truths or only lies. Alice says 'Bob is a liar.' Bob says 'Charlie is a liar.' Charlie says 'Alice and Bob are both liars.' Who is telling the truth and who is lying?",
        "In a certain country, 1/5 of men wear hats. 2/3 of men who don't wear hats have mustaches. 3/4 of men who wear hats also have mustaches. What fraction of men with mustaches wear hats?",
        "If all mammals have fur, and some animals with fur hibernate, must some mammals hibernate? Explain your answer with logical reasoning.",
        "If we know that 'if it rains, the ground gets wet' and we observe that 'the ground is not wet', what can we conclude about whether it rained?",
        "Three boxes are labeled 'Apples', 'Oranges', and 'Apples and Oranges'. All labels are incorrect. You can pick one fruit from one box. How can you correctly label all boxes with just this one observation?",
    ],
    "puzzle": [
        "You have 8 balls. One is slightly heavier than the others. Using a balance scale, how can you find the heavier ball with just 2 weighings?",
        "A farmer needs to cross a river with a fox, a chicken, and a sack of grain. The boat can only carry the farmer and one item at a time. If left alone, the fox will eat the chicken, and the chicken will eat the grain. How can the farmer get everything across safely?",
        "100 prisoners are in solitary cells, numbered 1 to 100. There's a central room with a switch. No prisoner can see the switch's position unless they are in the room. Each day, one random prisoner is taken to the central room and can choose to flip the switch or not. How can they determine when all 100 prisoners have visited the room at least once?",
        "You have 12 identical-looking coins. One coin is either heavier or lighter than the rest. Using a balance scale, how can you identify the odd coin and determine whether it's heavier or lighter in just 3 weighings?",
        "Five pirates have 100 gold coins to divide. They must vote on how to divide the coins. If at least 50% of the pirates agree on a plan, that's how the gold gets divided. Otherwise, the pirate proposing the plan is thrown overboard and the process starts over. Pirates are rational, greedy, and want to stay alive. What distribution should the first pirate propose?",
    ],
}

# Tracking variables
metrics = {}
responses = {}
active_requests = 0
start_time = None
running = True


def load_models_from_config():
    """Load available models from the configuration file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        models = {}

        # Check if we're using the new config format with total_number_of_jobs
        if "total_number_of_jobs" in config:
            total_jobs = config["total_number_of_jobs"]

            # Count the number of valid model entries
            model_indices = [k for k in config if isinstance(k, int)]
            model_count = len(model_indices)

            if model_count == 0:
                print(colorize("No valid model entries found in config", Colors.RED))
                sys.exit(1)

            # Calculate jobs per model (same logic as in slapi.py)
            base_jobs_per_model = total_jobs // model_count
            extra_jobs = total_jobs % model_count

            # Assign jobs to models
            for i, model_idx in enumerate(sorted(model_indices)):
                model_info = config[model_idx]
                model_name = model_info.get("model_name")

                if model_name:
                    # Calculate jobs for this model
                    if i < extra_jobs:
                        job_count = base_jobs_per_model + 1
                    else:
                        job_count = base_jobs_per_model

                    models[model_name] = {
                        "index": model_idx,
                        "jobs": job_count,
                    }

            print(
                colorize(
                    f"Using new config format: {total_jobs} total jobs distributed among {model_count} models",
                    Colors.CYAN,
                )
            )

        # Legacy format with per-model number_of_jobs
        else:
            for model_idx, model_info in config.items():
                if isinstance(model_idx, int) and "model_name" in model_info:
                    model_name = model_info.get("model_name")
                    models[model_name] = {
                        "index": model_idx,
                        "jobs": model_info.get("number_of_jobs", 1),
                    }

            print(
                colorize(
                    "Using legacy config format with per-model job counts",
                    Colors.YELLOW,
                )
            )

        return models
    except Exception as e:
        print(colorize(f"Error loading configuration: {e}", Colors.RED))
        sys.exit(1)


def shorten_model_name(name):
    """Shorten model name by taking first 4 chars of each part split by / or -"""
    # First split by '/'
    if "/" in name:
        parts = name.split("/")
        shortened_parts = []
        for part in parts:
            # Then split each part by '-' if needed
            if "-" in part:
                subparts = part.split("-")
                shortened_subparts = [
                    subpart[:4] if len(subpart) > 4 else subpart for subpart in subparts
                ]
                shortened_parts.append("-".join(shortened_subparts))
            else:
                shortened_parts.append(part[:4] if len(part) > 4 else part)
        return "/".join(shortened_parts)

    # If no '/', just split by '-'
    elif "-" in name:
        parts = name.split("-")
        shortened_parts = [part[:4] if len(part) > 4 else part for part in parts]
        return "-".join(shortened_parts)

    # No separators
    return name[:4] if len(name) > 4 else name


def initialize_metrics(models):
    """Initialize metrics tracking for all models and question types."""
    global metrics, responses

    for model_name in models:
        metrics[model_name] = {
            "count": 0,
            "success": 0,
            "latencies": [],
            "tokens_per_second": [],
            "max_latency_all_time": 0,  # Track all-time maximum latency
            "total_tokens": 0,  # Track total tokens generated
            "time_window_start": None,  # For measuring throughput in a time window
            "time_window_tokens": 0,  # Tokens generated in current time window
            "question_types": {},
        }
        responses[model_name] = {}

        for q_type in QUESTION_TYPES:
            metrics[model_name]["question_types"][q_type] = {
                "count": 0,
                "success": 0,
                "latencies": [],
                "tokens_per_second": [],
            }
            responses[model_name][q_type] = []


def get_random_question(question_type):
    """Get a random question from the specified type."""
    return random.choice(QUESTION_TYPES[question_type])


async def make_request(client, model, question_type):
    """Send a request to the model and track metrics."""
    global active_requests, metrics, responses

    # Get random question for this type
    question = get_random_question(question_type)

    # Increment active requests counter
    active_requests += 1

    request_start_time = time.time()
    success = False
    latency = 0
    output_tokens = 0
    response_text = ""

    try:
        response = await client.completions.create(
            model=model,
            prompt=question,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
        )

        end_time = time.time()
        latency = end_time - request_start_time
        response_text = response.choices[0].text.strip()
        output_tokens = len(response_text.split())  # Approximate
        success = True

        # Update overall metrics
        metrics[model]["count"] += 1
        metrics[model]["success"] += 1
        metrics[model]["latencies"].append(latency)

        # Track total tokens for aggregate throughput calculation
        metrics[model]["total_tokens"] += output_tokens

        # Initialize time window if not set
        if metrics[model]["time_window_start"] is None:
            metrics[model]["time_window_start"] = time.time()
            metrics[model]["time_window_tokens"] = 0

        # Update tokens in current time window
        metrics[model]["time_window_tokens"] += output_tokens

        # If time window is over 30 seconds, calculate tokens/sec and reset window
        time_in_window = time.time() - metrics[model]["time_window_start"]
        if time_in_window >= 30:  # 30-second rolling window
            tokens_per_second = metrics[model]["time_window_tokens"] / time_in_window
            metrics[model]["tokens_per_second"].append(tokens_per_second)

            # Reset window
            metrics[model]["time_window_start"] = time.time()
            metrics[model]["time_window_tokens"] = 0

            # Keep only the last 10 measurements
            if len(metrics[model]["tokens_per_second"]) > 10:
                metrics[model]["tokens_per_second"] = metrics[model][
                    "tokens_per_second"
                ][-10:]

        # Update the all-time maximum latency
        if latency > metrics[model]["max_latency_all_time"]:
            metrics[model]["max_latency_all_time"] = latency

        # Keep only last 100 latency samples for rolling average
        if len(metrics[model]["latencies"]) > 100:
            metrics[model]["latencies"] = metrics[model]["latencies"][-100:]

        # Update question type specific metrics
        metrics[model]["question_types"][question_type]["count"] += 1
        metrics[model]["question_types"][question_type]["success"] += 1
        metrics[model]["question_types"][question_type]["latencies"].append(latency)

        # Store example response (keeping only a limited number of samples)
        if len(responses[model][question_type]) < RESPONSE_SAMPLE_SIZE:
            responses[model][question_type].append(
                {"question": question, "response": response_text}
            )

    except Exception as e:
        end_time = time.time()
        latency = end_time - request_start_time
        metrics[model]["count"] += 1
        metrics[model]["question_types"][question_type]["count"] += 1
        # Don't increment success count

        # Even failed requests can update the max latency
        if latency > metrics[model]["max_latency_all_time"]:
            metrics[model]["max_latency_all_time"] = latency

    finally:
        # Decrement active requests counter
        active_requests -= 1

    return {
        "model": model,
        "question_type": question_type,
        "success": success,
        "latency": latency,
        "tokens": output_tokens,
        "response": response_text,
    }


def calculate_model_weights(model_names):
    """
    Calculate weights for model selection based on the imbalance ratio.

    Args:
        model_names: List of model names

    Returns:
        Dictionary mapping model names to their selection weights/probabilities
    """
    # If only one model, return 100% for that model
    if len(model_names) <= 1:
        return {model_names[0]: 1.0} if model_names else {}

    # If imbalance ratio is 0, distribute evenly
    if IMBALANCE_RATIO == 0:
        weight = 1.0 / len(model_names)
        return {name: weight for name in model_names}

    # Calculate exponential weights based on position and imbalance ratio
    # Later models in the list get progressively higher weights
    weights = []
    for i in range(len(model_names)):
        # Use exponential function to create imbalance
        # Higher ratio = more dramatic differences between models
        weight = math.exp(i * IMBALANCE_RATIO)
        weights.append(weight)

    # Normalize weights to sum to 1.0
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Create dictionary mapping model names to their weights
    return {model_names[i]: normalized_weights[i] for i in range(len(model_names))}


async def run_benchmark(models):
    """Run continuous benchmark across all models and question types."""
    global active_requests, start_time, running

    # Initialize the async client
    client = AsyncOpenAI(base_url=BASE_URL, api_key="not-needed")

    # Record start time
    start_time = time.time()

    # Initialize metrics for all models
    initialize_metrics(models)

    # Set up task queue
    model_names = list(models.keys())
    question_types = list(QUESTION_TYPES.keys())

    # Calculate model weights for selection based on imbalance ratio
    model_weights = calculate_model_weights(model_names)
    model_weights_list = [(model, weight) for model, weight in model_weights.items()]

    # Set up display task
    display_task = asyncio.create_task(display_metrics(models, UPDATE_INTERVAL))

    # Main benchmarking loop
    while running:
        # Generate new tasks up to MAX_CONCURRENCY
        while active_requests < MAX_CONCURRENCY and running:
            # Select model based on weights
            if IMBALANCE_RATIO > 0:
                # Weighted random selection using the calculated probabilities
                # Choose a random model according to the weights
                model = random.choices(
                    [item[0] for item in model_weights_list],
                    weights=[item[1] for item in model_weights_list],
                    k=1,
                )[0]
            else:
                # Original behavior: randomly select model
                model = random.choice(model_names)

            # Randomly select question type
            q_type = random.choice(question_types)

            # Create a task for this request
            asyncio.create_task(make_request(client, model, q_type))

            # Small sleep to avoid overwhelming the event loop
            await asyncio.sleep(0.01)

        # Wait a bit before checking again
        await asyncio.sleep(0.1)

    # Cancel the display task
    display_task.cancel()

    # Generate final report
    generate_final_report(models)


async def display_metrics(models, interval):
    """Periodically display metrics."""
    global metrics, start_time

    while True:
        # Clear screen (cross-platform)
        os.system("cls" if os.name == "nt" else "clear")

        # Get current timestamp and runtime
        now = datetime.now()
        runtime = time.time() - start_time
        runtime_str = str(timedelta(seconds=int(runtime)))

        # Print header
        print(colorize("=" * 100, Colors.HEADER))
        print(
            colorize(
                " vLLM Benchmark - Real-time Metrics ", Colors.HEADER + Colors.BOLD
            ).center(100, "=")
        )
        print(
            colorize(
                f" Runtime: {runtime_str} | Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')} ",
                Colors.CYAN,
            ).center(100, "-")
        )
        print(colorize("=" * 100, Colors.HEADER))

        # Generate summary table
        summary_table = []
        summary_headers = [
            "Model",
            "Requests",
            "% of Total",
            "Failed",
            "Avg Latency (s)",
            "Max Latency (s)",
            "99% Latency (s)",
            "Tokens/sec",
        ]

        # Calculate total requests across all models for percentage
        total_requests = sum(metrics[model_name]["count"] for model_name in metrics)

        # Collect metrics for all models
        for model_name in models.keys():
            if model_name in metrics:
                m = metrics[model_name]

                # Get model index and shortened name
                model_index = models[model_name]["index"]
                short_name = shorten_model_name(model_name)
                display_name = f"{model_index}:{short_name}"

                # Calculate summary metrics
                requests = m["count"]
                request_percentage = (
                    (requests / total_requests * 100) if total_requests > 0 else 0
                )
                failed = requests - m["success"]
                avg_latency = (
                    sum(m["latencies"]) / len(m["latencies"]) if m["latencies"] else 0
                )

                # Use the all-time max latency instead of just recent samples
                max_latency = m["max_latency_all_time"]

                # Calculate 99th percentile latency (top 1%)
                if len(m["latencies"]) > 0:
                    sorted_latencies = sorted(m["latencies"])
                    percentile_99_idx = max(0, int(len(sorted_latencies) * 0.99) - 1)
                    percentile_99_latency = sorted_latencies[percentile_99_idx]
                else:
                    percentile_99_latency = 0

                # Calculate current throughput
                # If we have token/sec measurements from our time windows, use their average
                if m["tokens_per_second"]:
                    avg_tokens_per_sec = sum(m["tokens_per_second"]) / len(
                        m["tokens_per_second"]
                    )
                # If we have no measurements yet but have a current window, calculate from that
                elif m["time_window_start"] is not None:
                    time_in_current_window = time.time() - m["time_window_start"]
                    if time_in_current_window > 0 and m["time_window_tokens"] > 0:
                        avg_tokens_per_sec = (
                            m["time_window_tokens"] / time_in_current_window
                        )
                    else:
                        avg_tokens_per_sec = 0
                # Fallback: calculate from total tokens and runtime if we have completed requests
                elif runtime > 0 and m["total_tokens"] > 0:
                    avg_tokens_per_sec = m["total_tokens"] / runtime
                else:
                    avg_tokens_per_sec = 0

                # Format failed count with color based on value
                if failed == 0:
                    failed_str = colorize("0", Colors.GREEN)
                elif failed <= requests * 0.05:  # Less than 5% failed
                    failed_str = colorize(f"{failed}", Colors.YELLOW)
                else:
                    failed_str = colorize(f"{failed}", Colors.RED)

                # Format percentages differently based on imbalance ratio
                if IMBALANCE_RATIO > 0:
                    # Highlight imbalanced request distribution
                    request_pct_str = colorize(
                        f"{request_percentage:.1f}%", Colors.YELLOW
                    )
                else:
                    request_pct_str = f"{request_percentage:.1f}%"

                summary_table.append(
                    [
                        display_name,
                        requests,
                        request_pct_str,
                        failed_str,
                        f"{avg_latency:.2f}",
                        f"{max_latency:.2f}",
                        f"{percentile_99_latency:.2f}",
                        f"{avg_tokens_per_sec:.1f}",
                    ]
                )

        # Sort table by model index
        summary_table.sort(key=lambda x: x[0])

        # Print summary table
        print(tabulate(summary_table, headers=summary_headers, tablefmt="pretty"))

        # Print active requests
        print(
            colorize(
                f"\nActive requests: {active_requests}/{MAX_CONCURRENCY}", Colors.CYAN
            )
        )

        # Print imbalance ratio info if set
        if IMBALANCE_RATIO > 0:
            print(
                colorize(
                    f"Imbalance ratio: {IMBALANCE_RATIO} (higher values = more imbalanced distribution)",
                    Colors.YELLOW,
                )
            )

        print(colorize("Press Ctrl+C to stop and generate final report", Colors.YELLOW))

        # Wait for next update
        await asyncio.sleep(interval)


def generate_final_report(models):
    """Generate and display final benchmark report."""
    global metrics, responses, start_time

    # Clear screen
    os.system("cls" if os.name == "nt" else "clear")

    # Calculate total runtime
    runtime = time.time() - start_time
    runtime_str = str(timedelta(seconds=int(runtime)))

    # Print header
    print(colorize("=" * 100, Colors.HEADER))
    print(
        colorize(" vLLM Benchmark - Final Report ", Colors.HEADER + Colors.BOLD).center(
            100, "="
        )
    )
    print(colorize(f" Total Runtime: {runtime_str} ", Colors.CYAN).center(100, "-"))
    print(colorize("=" * 100, Colors.HEADER))

    # Generate performance summary
    print(colorize("\nPerformance Summary:", Colors.BOLD))

    summary_table = []
    summary_headers = [
        "Model",
        "Requests",
        "% of Total",
        "Failed",
        "Avg Latency (s)",
        "Max Latency (s)",
        "99% Latency (s)",
        "Tokens/sec",
    ]

    # Calculate total requests across all models for percentage
    total_requests = sum(
        metrics[model_name]["count"] for model_name in metrics if model_name in metrics
    )

    # Collect metrics for all models
    for model_name in models.keys():
        if model_name in metrics:
            m = metrics[model_name]

            # Get model index and shortened name
            model_index = models[model_name]["index"]
            short_name = shorten_model_name(model_name)
            display_name = f"{model_index}:{short_name}"

            # Calculate summary metrics
            requests = m["count"]
            request_percentage = (
                (requests / total_requests * 100) if total_requests > 0 else 0
            )
            failed = requests - m["success"]
            avg_latency = (
                sum(m["latencies"]) / len(m["latencies"]) if m["latencies"] else 0
            )

            # Use the all-time max latency instead of just recent samples
            max_latency = m["max_latency_all_time"]

            # Calculate 99th percentile latency (top 1%)
            if len(m["latencies"]) > 0:
                sorted_latencies = sorted(m["latencies"])
                percentile_99_idx = max(0, int(len(sorted_latencies) * 0.99) - 1)
                percentile_99_latency = sorted_latencies[percentile_99_idx]
            else:
                percentile_99_latency = 0

            # Calculate overall tokens per second for the entire runtime
            if runtime > 0 and m["total_tokens"] > 0:
                avg_tokens_per_sec = m["total_tokens"] / runtime
            else:
                avg_tokens_per_sec = 0

            # Format failed count with color based on value
            if failed == 0:
                failed_str = colorize("0", Colors.GREEN)
            elif failed <= requests * 0.05:  # Less than 5% failed
                failed_str = colorize(f"{failed}", Colors.YELLOW)
            else:
                failed_str = colorize(f"{failed}", Colors.RED)

            # Format percentages differently based on imbalance ratio
            if IMBALANCE_RATIO > 0:
                # Highlight imbalanced request distribution
                request_pct_str = colorize(f"{request_percentage:.1f}%", Colors.YELLOW)
            else:
                request_pct_str = f"{request_percentage:.1f}%"

            summary_table.append(
                [
                    display_name,
                    requests,
                    request_pct_str,
                    failed_str,
                    f"{avg_latency:.2f}",
                    f"{max_latency:.2f}",
                    f"{percentile_99_latency:.2f}",
                    f"{avg_tokens_per_sec:.1f}",
                ]
            )

    # Sort table by model index
    summary_table.sort(key=lambda x: x[0])

    # Print summary table
    print(tabulate(summary_table, headers=summary_headers, tablefmt="pretty"))

    # Print imbalance ratio information if used
    if IMBALANCE_RATIO > 0:
        print(colorize(f"\nImbalance ratio: {IMBALANCE_RATIO}", Colors.YELLOW))
        print(
            "Higher ratios create more imbalanced request distribution between models"
        )
        print(
            "This simulates different request patterns to test dynamic load balancing"
        )

        # Show the theoretical model weights used
        model_weights = calculate_model_weights(list(models.keys()))
        print(colorize("\nTheoretical request distribution:", Colors.CYAN))
        weight_table = []
        for model_name, weight in model_weights.items():
            model_index = models[model_name]["index"]
            short_name = shorten_model_name(model_name)
            display_name = f"{model_index}:{short_name}"

            # Calculate actual percentage from data
            actual_count = metrics[model_name]["count"] if model_name in metrics else 0
            actual_pct = (
                (actual_count / total_requests * 100) if total_requests > 0 else 0
            )

            weight_table.append([display_name, f"{weight:.1%}", f"{actual_pct:.1f}%"])

        print(
            tabulate(
                weight_table,
                headers=["Model", "Theoretical %", "Actual %"],
                tablefmt="pretty",
            )
        )

    # Generate example responses for each model and question type
    print(colorize("\nExample Responses:", Colors.BOLD))

    for model_name in models.keys():
        # Get model index and shortened name for display
        model_index = models[model_name]["index"]
        short_name = shorten_model_name(model_name)
        display_name = f"{model_index}:{short_name}"

        print(colorize(f"\n{display_name}:", Colors.CYAN + Colors.BOLD))

        for q_type in QUESTION_TYPES:
            if (
                model_name in responses
                and q_type in responses[model_name]
                and responses[model_name][q_type]
            ):
                print(
                    colorize(f"\n  {q_type.replace('_', ' ').title()}:", Colors.YELLOW)
                )

                # Show one example response
                example = responses[model_name][q_type][0]
                question = example["question"]
                response = example["response"]

                print(colorize(f"    Q: {question}", Colors.BLUE))
                print(f"    A: {response}")


def signal_handler(sig, frame):
    """Handle Ctrl+C by setting running to False."""
    global running
    running = False
    print(
        colorize("\nStopping benchmark and generating final report...", Colors.YELLOW)
    )


def main():
    """Entry point to run the benchmark."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="vLLM Continuous Benchmark")
    parser.add_argument(
        "--imbalance-ratio",
        type=float,
        default=0.0,
        help="Imbalance ratio for request distribution (0 = even, higher = more imbalanced)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="API port to use for the benchmark",
    )
    args = parser.parse_args()

    global IMBALANCE_RATIO, BASE_URL
    IMBALANCE_RATIO = args.imbalance_ratio
    BASE_URL = f"http://localhost:{args.port}/v1"

    # Load models from config
    models = load_models_from_config()

    if not models:
        print(colorize("No models found in config.yaml", Colors.RED))
        return

    # Print startup banner
    print(colorize("=" * 100, Colors.HEADER))
    print(
        colorize(" vLLM Continuous Benchmark ", Colors.HEADER + Colors.BOLD).center(
            100, "="
        )
    )
    print(colorize("=" * 100, Colors.HEADER))
    print(colorize("Loaded models:", Colors.CYAN))

    for model_name, info in models.items():
        print(colorize(f"  - {model_name}", Colors.GREEN))

    print(colorize(f"\nQuestion types:", Colors.CYAN))
    for q_type in QUESTION_TYPES:
        print(colorize(f"  - {q_type.replace('_', ' ').title()}", Colors.GREEN))

    print(colorize(f"\nBenchmark settings:", Colors.CYAN))
    print(colorize(f"  - Max concurrency: {MAX_CONCURRENCY}", Colors.GREEN))
    print(colorize(f"  - Update interval: {UPDATE_INTERVAL} seconds", Colors.GREEN))
    print(colorize(f"  - API endpoint: {BASE_URL}", Colors.GREEN))
    print(colorize(f"  - Max tokens per response: {MAX_TOKENS}", Colors.GREEN))
    print(colorize(f"  - Imbalance ratio: {IMBALANCE_RATIO}", Colors.GREEN))

    # If imbalance ratio is set, show the distribution
    if IMBALANCE_RATIO > 0:
        model_weights = calculate_model_weights(list(models.keys()))
        print(colorize("\nRequest distribution:", Colors.CYAN))
        for model_name, weight in model_weights.items():
            model_index = models[model_name]["index"]
            short_name = shorten_model_name(model_name)
            display_name = f"{model_index}:{short_name}"
            print(colorize(f"  - {display_name}: {weight:.1%}", Colors.GREEN))

    print(colorize("\nStarting benchmark in 3 seconds...", Colors.YELLOW))
    time.sleep(3)

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Run the benchmark
    asyncio.run(run_benchmark(models))


if __name__ == "__main__":
    main()
