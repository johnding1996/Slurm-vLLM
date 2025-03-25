# vLLM Slurm Toolkit

A comprehensive toolkit for managing, serving, and benchmarking multiple vLLM models on a Slurm cluster.

## Features

- **Multi-Model Support**: Run multiple vLLM models simultaneously
- **Automatic Job Management**: Monitors and restarts jobs as needed
- **Smart Load Balancing**: Distributes requests across model replicas
- **Comprehensive Benchmarking**: Test models with various question types
- **Real-time Metrics**: Monitor performance with colorful, easy-to-read tables
- **Dynamic Job Allocation**: Automatically adjusts job distribution based on request patterns

## Quick Start

### 1. Installation

```bash
# Clone the repository (if you haven't already)
git clone <repository-url>
cd <repository-directory>

# Run the installation script
./install.sh
```

The installation script will:
- Check for required dependencies (Python 3.9-3.12, CUDA 12.4 only, Slurm)
- Create a virtual environment in `./venv`
- Install vLLM and other dependencies
- Make scripts executable

### 2. Configuration

Create or edit `config.yaml` to specify your models:

```yaml
# Dynamic balancing mode with automatic job distribution
total_number_of_jobs: 10

1:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
  gpu: "gpu:1"

2:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  gpu: "gpu:1"
```

### 3. Launch vLLM System

```bash
# Activate the virtual environment
source ./venv/bin/activate

# Start the vLLM Slurm API system
./slapi.py
```

This will:
- Submit jobs for each model specified in `config.yaml`
- Distribute jobs according to request patterns (if using dynamic balancing)
- Monitor job status and restart jobs if they fail
- Start the API proxy server that provides a unified endpoint for all models
- Display status tables showing running jobs and backend health

### 4. Run Benchmarks

```bash
# In a new terminal, activate the virtual environment
source ./venv/bin/activate

# Run the benchmarking tool
./bench.py
```

The benchmarking tool:
- Tests all models with various question types
- Displays real-time performance metrics
- Generates a comprehensive final report when stopped (Ctrl+C)

## Command Reference

### vLLM Slurm API System

```bash
./slapi.py [options]
```

Options:
- `--config FILE`: Path to configuration YAML file (default: `config.yaml`)
- `--port PORT`: Port to run the API proxy server on (default: 9090)
- `--job-prefix CHAR`: Single lowercase letter prefix for job names (default: 'v')
- `--monitor-interval SECONDS`: Interval between status checks (default: 5)
- `--cancel-all`: Cancel all vLLM jobs and exit
- `--test-mode`: Run in test mode with simulated job failures
- `--api-only`: Run only the API proxy without job management
- `--job-only`: Run only the job management without API proxy

### Benchmarking Tool

```bash
./bench.py [options]
```

Options:
- `--imbalance-ratio FLOAT`: Imbalance ratio for request distribution (0 = even, higher = more imbalanced)
- `--port PORT`: API port to use for the benchmark (default: 9090)

## Usage Scenarios

### Running the Complete System

```bash
# Run the complete system (job management + API proxy)
./slapi.py
```

### Running Only the Job Management

```bash
# Run only the job management component
./slapi.py --job-only
```

### Running Only the API Proxy

```bash
# Run only the API proxy component
./slapi.py --api-only
```

### Canceling All Jobs

```bash
# Cancel all vLLM jobs
./slapi.py --cancel-all
```

### Changing the Job Prefix

```bash
# Use a different job name prefix (e.g., 's' instead of 'v')
./slapi.py --job-prefix s
```

### Running with Custom Port

```bash
# Run the system on a custom port
./slapi.py --port 8080

# Run the benchmark against the custom port
./bench.py --port 8080
```

### Running the Benchmark with Imbalanced Requests

```bash
# Run the benchmark with an imbalance ratio of 1.0
# (models listed later in config.yaml receive more requests)
./bench.py --imbalance-ratio 1.0
```

## API Usage

The API proxy provides an OpenAI-compatible endpoint. Here are examples of how to use it:

### Async OpenAI Client Example

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    # Initialize async client with our endpoint
    client = AsyncOpenAI(base_url="http://localhost:9090/v1", api_key="not-needed")
    
    # Create a completion
    response = await client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",  # Use any model from config.yaml
        prompt="Explain quantum computing in simple terms",
        max_tokens=1000,
        temperature=0.7
    )
    
    print(response.choices[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

### Direct Async HTTP Request Example

```python
import asyncio
import httpx
import json

async def query_model():
    url = "http://localhost:9090/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt": "What are the main applications of deep learning?",
        "max_tokens": 500,
        "temperature": 0.5,
        "top_p": 0.9
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["text"]
        else:
            return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    result = asyncio.run(query_model())
    print(result)
```

## Dynamic Job Allocation

When using dynamic job allocation (`total_number_of_jobs` in config.yaml):

1. The system starts with an even distribution of jobs across all models
2. As requests come in, the system tracks which models are being used more frequently
3. During rebalancing, jobs are reallocated based on request patterns
4. Each model is guaranteed at least 2 jobs for redundancy

The Model Balance Status table shows the current and target job distribution for each model.

## Troubleshooting

- **Job submission fails**: Check Slurm configuration and queue availability
- **API proxy can't connect to backends**: Ensure jobs are running and ports are accessible
- **Benchmarking tool shows low success rate**: Check model health and API proxy logs

## Advanced Configuration

### Time Limits

By default, jobs in non-test mode are assigned random time limits between 30 minutes and 1 hour to prevent all jobs from timing out simultaneously. You can adjust these limits by modifying the `TIME_LIMIT_LOWER` and `TIME_LIMIT_UPPER` constants in `slapi.py`.

### Monitoring Intervals

You can adjust how frequently the system checks for job status and backend health using the `--monitor-interval` option. Lower values provide more responsive monitoring but may increase system load.

### Benchmark Tool Parameters

The benchmark tool offers several options to customize your testing:

- `--imbalance-ratio FLOAT`: Control the distribution of requests between models (0 = even, higher values = more imbalanced)
- `--port PORT`: Connect to a specific API port (default: 9090)
- `MAX_CONCURRENCY`: Constant that controls the maximum number of concurrent requests (default: 400)
- `UPDATE_INTERVAL`: Controls how frequently the metrics display updates (default: 2 seconds)
- `MAX_TOKENS`: Maximum number of tokens to generate per response (default: 1000)
- `RESPONSE_SAMPLE_SIZE`: Number of example responses to keep per model/question type (default: 2)

You can modify the benchmark's question types by editing the `QUESTION_TYPES` dictionary in `bench.py`.
