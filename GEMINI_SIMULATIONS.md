# Gemini Model Simulations

This document describes the full-scale simulations run with Google's Gemini 2.0 models.

## Models Used

1. **gemini-2.0-flash** - Google's latest fast model
   - Cost: $0.10/$0.40 per 1M tokens
   - Estimated simulation cost: $1.79

2. **gemini-2.0-flash-lite** - Lighter, more efficient variant
   - Cost: $0.075/$0.30 per 1M tokens  
   - Estimated simulation cost: $1.34

## Simulation Parameters

### Gruber & Saez (2002) Replication
- Income levels: 16 (from $50,000 to $200,000 in $10,000 steps)
- Tax rates: 11 (from 15% to 35% in 2% steps)
- Responses per scenario: 100
- Total API calls per model: 17,600

### PKNF (2024) Lab Experiment Replication  
- Treatments: 5 (different tax schedule combinations)
- Subjects per treatment: 100
- Rounds: 16
- Total API calls per model: 8,000

**Total API calls per model: 25,600**

## Running the Simulations

```bash
# Set your API key
export EXPECTED_PARROT_API_KEY=your-key-here

# Run all simulations
./run_gemini_simulations.sh
```

## Expected Output

Each simulation produces a CSV file in `book/data/`:
- `gruber_saez_results_gemini-2.0-flash.csv`
- `gruber_saez_results_gemini-2.0-flash-lite.csv`
- `pknf_results_gemini-2.0-flash.csv`
- `pknf_results_gemini-2.0-flash-lite.csv`

## Analysis

After simulations complete, we'll analyze:
1. ETI estimates by model
2. Consistency of responses
3. Model differences in tax understanding
4. Response patterns across income levels

## Cache Benefits

Thanks to EDSL's universal cache:
- First run: ~$3.13 total
- Subsequent runs: FREE
- Can re-analyze data without additional API costs