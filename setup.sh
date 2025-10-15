#!/bin/bash
# Setup script for Rationality LLM

echo "Setting up Rationality LLM..."

# Check Python version (requires 3.10+)
PYTHON_CMD=""
for cmd in python3.10 python3.11 python3.12 python3 python; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD=$cmd
            echo "Found Python $VERSION"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.10 or higher is required"
    echo "Please install Python 3.10+ and try again"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python integrated_verification.py"
echo "  python examples/basic_usage.py"
echo ""
echo "To use real LLM verification, set your API key:"
echo "  export ANTHROPIC_API_KEY='your-key'  # For Claude"
echo "  export OPENAI_API_KEY='your-key'     # For GPT"
