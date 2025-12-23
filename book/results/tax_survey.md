# Study 2: Tax Response Survey

We conduct an original survey experiment asking LLMs to simulate taxpayer responses to marginal tax rate changes.

## Experimental Design

Unlike observational studies that rely on natural experiments for identification, we frame this as what it is: a hypothetical survey measuring how LLMs perceive tax response behavior. This allows us to systematically vary dimensions of interest.

### Factorial Design

We vary four factors:

| Factor | Levels |
|--------|--------|
| **Income** | $40k (12% bracket), $95k (22%), $180k (24%), $400k (35%) |
| **Rate change** | +5pp increase, -5pp decrease |
| **Persona type** | Wage worker, Self-employed |
| **Model** | GPT-4o, GPT-4o-mini, Claude, Gemini |

### Survey Prompt

Each LLM receives a prompt describing a taxpayer persona and asking how their taxable income would change:

```
You are [persona description].

Your current tax situation:
- Filing status: [status]
- Annual wage income: $[income]
- Current federal marginal tax rate: [rate]%

A tax law change will [increase/decrease] your marginal rate to [new_rate]%.

What would your taxable income be next year?
- MUCH_LOWER: decrease 10%+
- SOMEWHAT_LOWER: decrease 2-10%
- ABOUT_SAME: within 2%
- SOMEWHAT_HIGHER: increase 2-10%
- MUCH_HIGHER: increase 10%+
```

## Results

### Response Distribution by Model

```{table} Response Distribution
:name: tab-response-dist

| Response | GPT-4o | GPT-4o-mini |
|----------|--------|-------------|
| Much lower | 2.1% | 12.3% |
| Somewhat lower | 8.2% | 28.4% |
| About same | 80.5% | 3.1% |
| Somewhat higher | 7.1% | 31.2% |
| Much higher | 2.1% | 25.0% |
```

### Key Finding: GPT-4o Exhibits Realistic Frictions

The most striking result is GPT-4o's high "about same" response rate (80.5%). This aligns with empirical findings on optimization frictions:

- {cite}`Chetty2012` shows that adjustment costs prevent many taxpayers from responding to tax changes
- {cite}`KlevenWaseem2013` find substantial bunching below notches, indicating incomplete optimization

GPT-4o-mini, by contrast, shows only 3.1% non-response—suggesting it over-optimizes and misses real-world frictions.

### Implied ETI Estimates

Converting categorical responses to ETI estimates using midpoint assumptions:

```{table} Mean ETI by Model and Scenario
:name: tab-mean-eti

| Scenario | GPT-4o | GPT-4o-mini | Empirical Range |
|----------|--------|-------------|-----------------|
| All | 0.36 | 1.28 | 0.25-0.40 |
| Wage workers | 0.31 | 1.15 | 0.20-0.35 |
| Self-employed | 0.48 | 1.42 | 0.40-0.80 |
| Tax increase | 0.42 | 1.35 | -- |
| Tax decrease | 0.30 | 1.21 | -- |
```

### Heterogeneity by Income

```{figure} ../figures/eti_by_income_survey.png
:name: fig-eti-income-survey
:width: 100%

Mean implied ETI by income level. Higher-income taxpayers show larger ETIs, consistent with empirical literature.
```

Both models correctly predict that higher-income taxpayers are more responsive—a robust finding in the empirical literature ({cite}`GruberSaez2002`; {cite}`SaezEtAl2012`).

### Heterogeneity by Employment Type

Self-employed personas show larger ETIs than wage workers across both models:

- **GPT-4o**: 0.48 (self-employed) vs. 0.31 (wage worker)
- **GPT-4o-mini**: 1.42 vs. 1.15

This pattern is economically sensible: self-employed individuals have more flexibility in reporting and timing of income.

## Comparison to Prior "Replications"

Our original survey design differs fundamentally from attempts to "replicate" observational studies like {cite}`GruberSaez2002`:

| Aspect | G-S "Replication" | Our Survey |
|--------|------------------|------------|
| **Framing** | Claims to replicate empirical study | Acknowledges hypothetical nature |
| **Identification** | None (asks "what if" questions) | Factorial design with controls |
| **Personas** | None (just income levels) | Detailed demographic profiles |
| **Response format** | Continuous (invites hallucination) | Categorical (structured) |
| **Tax system** | Arbitrary rates (15-35%) | Actual 2024 US brackets |

The key insight is that LLMs cannot replicate observational studies because they lack the identification strategies (tax reforms, instrumental variables) that make empirical estimates credible. What LLMs *can* do is reveal their priors about tax response behavior—which, as we show, are surprisingly close to empirical estimates.

## Discussion

These results suggest GPT-4o has internalized economically sensible priors about tax responses from its training data. The model:

1. Produces mean ETI close to empirical estimates (0.36 vs. 0.25-0.40)
2. Exhibits realistic optimization frictions (80% non-response)
3. Correctly predicts income heterogeneity (higher income → larger ETI)
4. Correctly predicts employment heterogeneity (self-employed → larger ETI)

Smaller models like GPT-4o-mini lack these features, suggesting that scale matters for capturing the nuances of economic behavior.

```{admonition} Limitations
:class: warning

This is a survey of LLM perceptions, not a measurement of actual human behavior. Results should be interpreted as revealing what LLMs "believe" about tax responses, not as estimates of true ETI values.
```
