# vLLM Slurm Toolkit

A comprehensive toolkit for managing, serving, and benchmarking multiple vLLM models on a Slurm cluster.

## Features

- **Multi-Model Support**: Run multiple vLLM models simultaneously
- **Automatic Job Management**: Monitors and restarts jobs as needed
- **Smart Load Balancing**: Distributes requests across model replicas
- **Comprehensive Benchmarking**: Test models with various question types
- **Real-time Metrics**: Monitor performance with colorful, easy-to-read tables

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
1:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
  number_of_jobs: 2
  gpu: "gpu:1"

2:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  number_of_jobs: 2
  gpu: "gpu:1"
```

### 3. Launch vLLM Jobs

```bash
# Activate the virtual environment
source ./venv/bin/activate

# Start the Slurm job manager
./slurm.py
```

This will:
- Submit jobs for each model specified in `config.yaml`
- Monitor job status and restart jobs if they fail
- Display a status table showing all running jobs

### 4. Start the API Proxy

```bash
# In a new terminal, activate the virtual environment
source ./venv/bin/activate

# Start the API proxy
./api.py
```

The API proxy:
- Merges all model endpoints into a single OpenAI-compatible API
- Implements smart load balancing across model replicas
- Monitors backend health and routes requests accordingly

### 5. Run Benchmarks

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

### Slurm Job Manager

```bash
./slurm.py [options]
```

Options:
- `--config FILE`: Path to configuration YAML file (default: `config.yaml`)
- `--monitor-interval SECONDS`: Interval between job status checks (default: 30)
- `--cancel-all`: Cancel all vLLM jobs and exit

### API Proxy

```bash
./api.py [options]
```

Options:
- `--port PORT`: Port to run the API proxy server on (default: 9090)
- `--monitor-interval SECONDS`: Interval between backend checks (default: 5)

### Benchmarking Tool

```bash
./bench.py
```

The benchmarking tool runs continuously until interrupted with Ctrl+C.

## API Usage

The API proxy provides an OpenAI-compatible endpoint at `http://localhost:9090/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9090/v1", api_key="not-needed")

response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",  # Use any model from config.yaml
    prompt="Explain quantum computing in simple terms",
    max_tokens=1000,
    temperature=0.7
)

print(response.choices[0].text)
```

## Troubleshooting

- **Job submission fails**: Check Slurm configuration and queue availability
- **API proxy can't connect to backends**: Ensure jobs are running and ports are accessible
- **Benchmarking tool shows low success rate**: Check model health and API proxy logs

## Advanced Usage

### Canceling All Jobs

```bash
./slurm.py --cancel-all
```

### Changing Log Directory

Logs are stored in `./vllm_logs` by default. You can modify this in `slurm.py`.

### Custom Question Types

Edit the `QUESTION_TYPES` dictionary in `bench.py` to add your own question categories. 
