#!/bin/bash
# Run simulations with Gemini 2.5 models using smaller batch sizes for reliability

echo "Running simulations with Gemini 2.5 models (safe mode)"
echo "====================================================="
echo ""

# Check for API key
if [ -z "$EXPECTED_PARROT_API_KEY" ]; then
    echo "Error: EXPECTED_PARROT_API_KEY environment variable not set"
    exit 1
fi

# Create data directory
mkdir -p book/data

# Run in test mode which uses fewer responses
echo "1. Running Gruber & Saez simulation with gemini-2.5-flash (test mode)..."
echo "------------------------------------------------------------------------"
uv run python book/scripts/run_gruber_saez_simulation.py --test --model gemini-2.5-flash --cache-analysis

echo ""
echo "2. Running Gruber & Saez simulation with gemini-2.5-flash-lite (test mode)..."
echo "-----------------------------------------------------------------------------"
uv run python book/scripts/run_gruber_saez_simulation.py --test --model gemini-2.5-flash-lite --cache-analysis

# Run PKNF simulations in test mode
echo ""
echo "3. Running PKNF simulation with gemini-2.5-flash (test mode)..."
echo "---------------------------------------------------------------"
uv run python book/scripts/run_pknf_simulation.py --test --model gemini-2.5-flash --cache-analysis

echo ""
echo "4. Running PKNF simulation with gemini-2.5-flash-lite (test mode)..."
echo "--------------------------------------------------------------------"
uv run python book/scripts/run_pknf_simulation.py --test --model gemini-2.5-flash-lite --cache-analysis

echo ""
echo "====================================================="
echo "All simulations complete!"
echo ""

# List generated files
echo "Generated data files:"
ls -la book/data/*gemini*.csv

echo ""
echo "Note: Running in test mode with reduced data:"
echo "- Gruber & Saez: 5 responses per rate (vs 100 in production)"
echo "- PKNF: 10 subjects per treatment (vs 100 in production)"
echo "Use --production flag for full simulations"