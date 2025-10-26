#!/bin/bash
# Setup script for Content Analysis Pipeline

set -e  # Exit on error

echo "=================================="
echo "Content Analysis Pipeline Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install numpy pandas matplotlib PyYAML pytest pytest-cov

echo "✓ Dependencies installed"

# Create necessary directories
echo "Creating directories..."
mkdir -p output
mkdir -p logs
mkdir -p tests
echo "✓ Directories created"

# Check if config file exists
if [ ! -f "config/pipeline_config.yaml" ]; then
    echo "⚠ Warning: Config file not found at config/pipeline_config.yaml"
    echo "  Please ensure the config file is in place before running the pipeline."
else
    echo "✓ Config file found"
fi

# Run tests if available
if [ -d "tests" ] && [ "$(ls -A tests/*.py 2>/dev/null)" ]; then
    echo ""
    echo "Running tests..."
    pytest tests/ -v
    echo "✓ Tests passed"
else
    echo "⚠ No tests found in tests/ directory"
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python run_pipeline.py"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
