# LLM-ETI: Experimental Design (v2)

## Summary of Changes from v1

### What We're Dropping
- **"Gruber-Saez Replication"** - This was never a replication. G-S used panel data with a real identification strategy. We were just asking hypothetical questions.

### What We're Keeping
- **PKNF Lab Experiment Replication** - This is valid. Same experimental design, LLM subjects instead of humans.

### What We're Adding
- **Original Tax Response Survey Study** - Properly designed factorial experiment with realistic personas.
- **Multi-Model Comparison** - Systematic comparison across model families (GPT, Claude, Gemini).

---

## Study 1: PKNF Lab Experiment Replication (Keep)

This study is methodologically sound. We replicate the exact experimental protocol:
- 16 rounds, tax reform at round 8
- Same tax schedules (Flat 25%, Flat 50%, Progressive with notch)
- Same randomization of labor endowments

**Key improvement needed:** Better prompt engineering based on exact PKNF instructions to human subjects.

---

## Study 2: Tax Response Survey (NEW - Replaces G-S "Replication")

### Design Principles
1. **Not a replication** - An original study asking: How do LLMs simulate tax responses?
2. **Realistic personas** - Based on CPS microdata demographics
3. **Actual tax system** - 2024 US federal brackets, not arbitrary rates
4. **Factorial design** - Systematic variation of key dimensions
5. **Within-subject** - Same persona faces multiple scenarios for consistency

### Factorial Design

**Factor 1: Income Level (4 levels)**
- $40,000 (12% bracket)
- $95,000 (22% bracket)
- $180,000 (24% bracket)
- $400,000 (35% bracket)

**Factor 2: Rate Change Direction (2 levels)**
- Increase (+5 percentage points)
- Decrease (-5 percentage points)

**Factor 3: Persona Type (4 levels)**
- Single software engineer, no dependents
- Married teacher with 2 children
- Self-employed contractor
- Retired pensioner with investment income

**Factor 4: Response Margin (2 levels)**
- Effort margin (work more/less)
- Reporting margin (deductions, timing)

**Total cells:** 4 × 2 × 4 × 2 = 64 unique scenarios
**Repetitions:** 50 per cell = 3,200 observations per model

### Prompt Template

```
You are [PERSONA_DESCRIPTION].

Your situation:
- Filing status: [single/married filing jointly]
- Annual wage income: $[INCOME]
- Other income: $[OTHER_INCOME]
- Current federal marginal tax rate: [CURRENT_RATE]%

A tax law change will [increase/decrease] your marginal rate to [NEW_RATE]%.

Consider how this might affect your:
1. Work effort (hours, overtime, second job)
2. Tax planning (timing of income, deductions, retirement contributions)

Question: What would be your taxable income next year relative to this year?
- Much lower (down 10%+)
- Somewhat lower (down 2-10%)
- About the same (within 2%)
- Somewhat higher (up 2-10%)
- Much higher (up 10%+)

Also provide a brief explanation of your reasoning.
```

### Why This Is Better

| Aspect | Old Approach | New Approach |
|--------|--------------|--------------|
| Framing | "Replication" of empirical study | Original survey experiment |
| Personas | None - just income levels | Realistic demographics |
| Tax rates | Arbitrary 15-35% grid | Actual 2024 US brackets |
| Response format | Continuous (invite hallucination) | Categorical + explanation |
| Identification | None | Factorial design |

---

## Study 3: Multi-Model Comparison

### Models to Test
1. **GPT-4o** (flagship)
2. **GPT-4o-mini** (cost-efficient)
3. **Claude Sonnet 4** (reasoning-focused)
4. **Gemini 2.5 Flash** (Google)
5. **Llama 3.1 70B** (open source)

### Research Questions
1. Do different model families produce systematically different ETI estimates?
2. Which models show more realistic optimization frictions?
3. Are there model-specific biases (always respond vs. never respond)?

---

## Data Sources for Personas

### CPS ASEC Variables
- `INCWAGE` - Wage income
- `INCDIVID` - Dividend income
- `INCINT` - Interest income
- `MARSTAT` - Marital status
- `NCHILD` - Number of children
- `OCC` - Occupation
- `CLASSWKR` - Self-employed vs. wage worker

We'll create 100 distinct personas sampled from CPS to ensure demographic realism.

---

## Hypotheses

### H1: Mean ETI Alignment
LLM-generated mean ETI will fall within the empirical range (0.2-0.5) for flagship models.

### H2: Income Heterogeneity
Higher-income personas will show larger ETI (consistent with empirical literature).

### H3: Margin Heterogeneity
Self-employed personas will show larger ETI than wage workers (real behavioral response).

### H4: Model Differences
Smaller/cheaper models will show less realistic frictions (higher response rates).

### H5: Direction Asymmetry
Tax increases will elicit larger responses than equivalent decreases (loss aversion).

---

## Power Analysis

With 50 observations per cell and 64 cells per model:
- 3,200 observations per model
- 16,000 observations across 5 models
- Power to detect effect size d=0.3 at α=0.05: >95%

---

## Timeline

1. **Phase 1:** Implement test suite and refactored code (TDD)
2. **Phase 2:** Run PKNF replication with improved prompts
3. **Phase 3:** Run tax response survey across models
4. **Phase 4:** Analysis and paper writing
5. **Phase 5:** Submit for (simulated) peer review
