#!/bin/bash

# ANSI color codes for colorful output
HEADER="\033[95m"
BLUE="\033[94m"
CYAN="\033[96m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
ENDC="\033[0m"
BOLD="\033[1m"
UNDERLINE="\033[4m"

# Function to print colorful messages
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${ENDC}"
}

# Function to print section headers
print_header() {
    local message=$1
    echo
    print_color "${HEADER}${BOLD}" "==================================================="
    print_color "${HEADER}${BOLD}" " ${message}"
    print_color "${HEADER}${BOLD}" "==================================================="
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if ! command_exists python3; then
        print_color "${RED}" "❌ Python 3 is not installed. Please install Python 3.9 or higher."
        exit 1
    fi

    # Get Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    python_major=$(echo $python_version | cut -d. -f1)
    python_minor=$(echo $python_version | cut -d. -f2)

    print_color "${CYAN}" "📌 Detected Python version: ${python_version}"

    # Check if Python version is 3.9-3.12
    if [ "$python_major" -ne 3 ] || [ "$python_minor" -lt 9 ] || [ "$python_minor" -gt 12 ]; then
        print_color "${YELLOW}" "⚠️ Warning: Recommended Python version is 3.9-3.12. You have ${python_version}."
        print_color "${YELLOW}" "⚠️ The installation may not work correctly."
        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_color "${RED}" "❌ Installation aborted."
            exit 1
        fi
    else
        print_color "${GREEN}" "✅ Python version check passed."
    fi
}

# Function to check CUDA version
check_cuda_version() {
    if ! command_exists nvcc; then
        print_color "${YELLOW}" "⚠️ NVCC not found. Checking for CUDA in other ways..."
        
        # Try nvidia-smi
        if command_exists nvidia-smi; then
            cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1,2)
            print_color "${CYAN}" "📌 Detected CUDA version (from driver): ${cuda_version}"
            
            # Check if CUDA version is at least 12.0
            cuda_major=$(echo $cuda_version | cut -d. -f1)
            cuda_minor=$(echo $cuda_version | cut -d. -f2)
            
            if [ "$cuda_major" -lt 12 ]; then
                print_color "${YELLOW}" "⚠️ Warning: Recommended CUDA version is 12.4 or higher. You have ${cuda_version}."
                print_color "${YELLOW}" "⚠️ vLLM may not work correctly with this CUDA version."
                read -p "Do you want to continue anyway? (y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    print_color "${RED}" "❌ Installation aborted."
                    exit 1
                fi
            else
                print_color "${GREEN}" "✅ CUDA version check passed."
            fi
        else
            print_color "${YELLOW}" "⚠️ Cannot determine CUDA version. vLLM requires CUDA 12.0 or higher."
            read -p "Do you want to continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_color "${RED}" "❌ Installation aborted."
                exit 1
            fi
        fi
    else
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1)
        print_color "${CYAN}" "📌 Detected CUDA version: ${cuda_version}"
        
        # Check if CUDA version is at least 12.0
        cuda_major=$(echo $cuda_version | cut -d. -f1)
        cuda_minor=$(echo $cuda_version | cut -d. -f2)
        
        if [ "$cuda_major" -lt 12 ]; then
            print_color "${YELLOW}" "⚠️ Warning: Recommended CUDA version is 12.4 or higher. You have ${cuda_version}."
            print_color "${YELLOW}" "⚠️ vLLM may not work correctly with this CUDA version."
            read -p "Do you want to continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_color "${RED}" "❌ Installation aborted."
                exit 1
            fi
        else
            print_color "${GREEN}" "✅ CUDA version check passed."
        fi
    fi
}

# Function to check if Slurm is available
check_slurm() {
    if ! command_exists sinfo; then
        print_color "${RED}" "❌ Slurm is not installed or not in PATH. This script requires Slurm."
        exit 1
    fi
    
    print_color "${GREEN}" "✅ Slurm is available."
}

# Function to create and activate virtual environment
create_venv() {
    print_color "${CYAN}" "📦 Creating virtual environment in ./venv..."
    
    # Check if venv already exists
    if [ -d "./venv" ]; then
        print_color "${YELLOW}" "⚠️ Virtual environment already exists."
        read -p "Do you want to recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_color "${YELLOW}" "🗑️ Removing existing virtual environment..."
            rm -rf ./venv
        else
            print_color "${CYAN}" "📌 Using existing virtual environment."
            return
        fi
    fi
    
    # Create virtual environment
    python3 -m venv ./venv
    
    if [ $? -ne 0 ]; then
        print_color "${RED}" "❌ Failed to create virtual environment."
        exit 1
    fi
    
    print_color "${GREEN}" "✅ Virtual environment created successfully."
}

# Function to install dependencies
install_dependencies() {
    print_color "${CYAN}" "📦 Installing dependencies..."
    
    # Activate virtual environment
    source ./venv/bin/activate
    
    # Upgrade pip
    print_color "${CYAN}" "📌 Upgrading pip..."
    pip install --upgrade pip
    
    # Install vLLM
    print_color "${CYAN}" "📌 Installing vLLM..."
    pip install vllm
    
    # Extract required packages from Python files
    print_color "${CYAN}" "📌 Installing additional dependencies..."
    
    # Install packages required by our scripts
    pip install pyyaml tabulate fastapi uvicorn httpx pydantic aiohttp
    
    # Deactivate virtual environment
    deactivate
    
    print_color "${GREEN}" "✅ Dependencies installed successfully."
}

# Function to create log directory
create_log_dir() {
    print_color "${CYAN}" "📁 Creating log directory..."
    
    if [ ! -d "./vllm_logs" ]; then
        mkdir -p ./vllm_logs
        print_color "${GREEN}" "✅ Log directory created."
    else
        print_color "${GREEN}" "✅ Log directory already exists."
    fi
}

# Function to make scripts executable
make_scripts_executable() {
    print_color "${CYAN}" "🔧 Making scripts executable..."
    
    chmod +x slapi.py
    chmod +x bench.py
    
    print_color "${GREEN}" "✅ Scripts are now executable."
}

# Main installation process
print_header "vLLM Slurm Toolkit Installer"

# Check requirements
print_color "${CYAN}" "🔍 Checking requirements..."
check_python_version
check_cuda_version
check_slurm

# Create virtual environment
print_header "Setting up Environment"
create_venv

# Install dependencies
install_dependencies

# Create log directory
create_log_dir

# Make scripts executable
make_scripts_executable

# Final message
print_header "Installation Complete"
print_color "${GREEN}" "✅ vLLM Slurm Toolkit has been successfully installed!"
print_color "${CYAN}" "📌 To use the toolkit:"
echo -e "   1. Activate the virtual environment: ${YELLOW}source ./venv/bin/activate${ENDC}"
echo -e "   2. Run the vLLM Slurm API system: ${YELLOW}./slapi.py${ENDC}"
echo -e "   3. Run the benchmarking tool: ${YELLOW}./bench.py${ENDC}"
print_color "${CYAN}" "📌 For more information, see the README.md file."
echo
print_color "${GREEN}${BOLD}" "Happy benchmarking! 🚀" 