#!/bin/bash
# Run full simulations with Gemini 2.5 models

echo "Running full simulations with Gemini 2.5 models"
echo "=============================================="
echo ""

# Check for API key
if [ -z "$EXPECTED_PARROT_API_KEY" ]; then
    echo "Error: EXPECTED_PARROT_API_KEY environment variable not set"
    exit 1
fi

# Create data directory
mkdir -p book/data

# Run Gruber & Saez simulations
echo "1. Running Gruber & Saez simulation with gemini-2.5-flash..."
echo "-------------------------------------------------------------"
uv run python book/scripts/run_gruber_saez_simulation.py --production --model gemini-2.5-flash --cache-analysis

echo ""
echo "2. Running Gruber & Saez simulation with gemini-2.5-flash-lite..."
echo "-----------------------------------------------------------------"
uv run python book/scripts/run_gruber_saez_simulation.py --production --model gemini-2.5-flash-lite --cache-analysis

# Run PKNF simulations
echo ""
echo "3. Running PKNF simulation with gemini-2.5-flash..."
echo "---------------------------------------------------"
uv run python book/scripts/run_pknf_simulation.py --production --model gemini-2.5-flash --cache-analysis

echo ""
echo "4. Running PKNF simulation with gemini-2.5-flash-lite..."
echo "-------------------------------------------------------"
uv run python book/scripts/run_pknf_simulation.py --production --model gemini-2.5-flash-lite --cache-analysis

echo ""
echo "=============================================="
echo "All simulations complete!"
echo ""

# List generated files
echo "Generated data files:"
ls -la book/data/*gemini*.csv

# Total cost estimate
echo ""
echo "Estimated costs:"
echo "- gemini-2.5-flash: ~$1.79 total"
echo "- gemini-2.5-flash-lite: ~$1.34 total"
echo "Total: ~$3.13"
echo ""
echo "Note: Subsequent runs are FREE due to EDSL's universal cache!"