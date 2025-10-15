# Supported Models Reference

Complete list of all models supported by the evaluation framework.

## Quick Comparison Command

```bash
# Compare GPT-4o, Claude, and Gemini (top picks)
python run_comparison.py --limit 20 --models gpt4o claude gemini
```

## All Supported Models

### OpenAI Models

| Model Key | Full Name | Model ID | Best For | Speed | Cost |
|-----------|-----------|----------|----------|-------|------|
| `gpt4` | GPT-4 | gpt-4 | Accuracy, reasoning | ⚡⚡ | $$$ |
| `gpt4o` | GPT-4o | gpt-4o | Latest, multimodal | ⚡⚡⚡ | $$ |
| `gpt4-turbo` | GPT-4 Turbo | gpt-4-turbo | Speed + capability | ⚡⚡⚡ | $$ |

**Setup**: `OPENAI_API_KEY` in .env

### Anthropic Claude Models

| Model Key | Full Name | Model ID | Best For | Speed | Cost |
|-----------|-----------|----------|----------|-------|------|
| `claude` | Claude 3.5 Sonnet | claude-3-5-sonnet-20241022 | Best balance | ⚡⚡⚡ | $$ |
| `claude-opus` | Claude 3 Opus | claude-3-opus-20240229 | Maximum capability | ⚡ | $$$$ |
| `claude-haiku` | Claude 3 Haiku | claude-3-haiku-20240307 | Fastest, cheapest | ⚡⚡⚡⚡ | $ |

**Setup**: `ANTHROPIC_API_KEY` in .env

### Google Gemini Models

| Model Key | Full Name | Model ID | Best For | Speed | Cost |
|-----------|-----------|----------|----------|-------|------|
| `gemini` | Gemini 2.0 Flash | gemini-2.0-flash-exp | Speed, low cost | ⚡⚡⚡⚡ | $ |
| `gemini-pro` | Gemini 2.5 Pro | gemini-2.5-pro-002 | Google's best | ⚡⚡ | $$ |

**Setup**: `GEMINI_API_KEY` in .env (see GEMINI_SETUP.md)

## Usage Examples

### Single Model Test
```bash
# Test just Gemini
python run_comparison.py --limit 20 --models gemini
```

### Head-to-Head Comparison
```bash
# GPT-4o vs Claude
python run_comparison.py --limit 20 --models gpt4o claude

# GPT-4o vs Gemini
python run_comparison.py --limit 20 --models gpt4o gemini

# All top models
python run_comparison.py --limit 20 --models gpt4o claude gemini
```

### Comprehensive Test
```bash
# Test all models (expensive!)
python run_comparison.py --limit 20 --models gpt4 gpt4o gpt4-turbo claude claude-opus claude-haiku gemini gemini-pro
```

### Budget-Friendly Test
```bash
# Cheapest models only
python run_comparison.py --limit 20 --models claude-haiku gemini
```

## Cost Estimates (20 examples)

| Model(s) | Total Cost | Per Example |
|----------|------------|-------------|
| gemini | ~$0.04 | $0.002 |
| claude-haiku | ~$0.06 | $0.003 |
| gemini-pro | ~$0.10 | $0.005 |
| gpt4o | ~$0.20 | $0.010 |
| claude | ~$0.20 | $0.010 |
| gpt4 | ~$0.40 | $0.020 |
| claude-opus | ~$0.60 | $0.030 |

**Note**: Costs are approximate and vary based on prompt length.

## Performance Characteristics

### Speed Ranking (Fastest to Slowest)
1. ⚡⚡⚡⚡ Gemini 2.0 Flash
2. ⚡⚡⚡⚡ Claude Haiku
3. ⚡⚡⚡ GPT-4o
4. ⚡⚡⚡ GPT-4 Turbo
5. ⚡⚡⚡ Claude Sonnet
6. ⚡⚡ GPT-4
7. ⚡⚡ Gemini 2.5 Pro
8. ⚡ Claude Opus

### Capability Ranking (Based on general benchmarks)
1. GPT-4o (Latest, strong reasoning)
2. Claude 3.5 Sonnet (Excellent reasoning)
3. Claude 3 Opus (Maximum capability)
4. Gemini 2.5 Pro (Google's best)
5. GPT-4 (Proven performance)
6. GPT-4 Turbo (Fast + capable)
7. Gemini 2.0 Flash (Fast + capable)
8. Claude Haiku (Speed optimized)

### Cost Efficiency (Performance per dollar)
1. Gemini 2.0 Flash
2. Claude Haiku
3. GPT-4o
4. Claude Sonnet
5. Gemini 2.5 Pro
6. GPT-4 Turbo
7. GPT-4
8. Claude Opus

## Your Previous Results (20 examples)

From your last run:

| Model | Accuracy | F1 Score | Speed (s/example) | Winner |
|-------|----------|----------|-------------------|--------|
| GPT-4 | 0.55 | 0.47 | 101s | 🏆 Best F1 |
| Claude Sonnet | 0.45 | 0.27 | 1611s | - |

**Insight**: GPT-4 performed better on contradiction detection, but took 16x longer than expected (possible API issues).

## Recommended Combinations

### Quick Iteration
```bash
# Fast + cheap for rapid testing
python run_comparison.py --limit 10 --models gemini claude-haiku
```

### Quality Comparison
```bash
# Best models from each provider
python run_comparison.py --limit 20 --models gpt4o claude gemini-pro
```

### Budget-Conscious
```bash
# Get good results without breaking the bank
python run_comparison.py --limit 50 --models gemini claude
```

### Comprehensive Analysis
```bash
# All top-tier models
python run_comparison.py --limit 20 --models gpt4 gpt4o claude claude-opus gemini-pro
```

## Adding Your API Keys

Edit `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=sk-proj-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-api03-...

# Google Gemini
GEMINI_API_KEY=AIzaSy...
```

## Troubleshooting

### Model Not Running?
- Check API key is set in .env
- Verify key has correct permissions
- Check API quota/limits

### Slow Performance?
- Try faster models (gemini, claude-haiku, gpt4o)
- Reduce --limit to fewer examples
- Check internet connection

### High Costs?
- Start with --limit 10
- Use cheaper models first (gemini, claude-haiku)
- Monitor costs in provider dashboards

## Next Steps

1. **Add API Keys**: Update `.env` with your keys
2. **Quick Test**: `python run_comparison.py --limit 5 --models gemini`
3. **Full Comparison**: `python run_comparison.py --limit 20 --models gpt4o claude gemini`
4. **Analyze**: Check results in `evaluation/reports/`

Ready to run! 🚀
