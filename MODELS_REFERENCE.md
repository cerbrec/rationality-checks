# Supported Models Reference

Complete list of all models supported by the evaluation framework.

## Quick Comparison Command

```bash
# Recommended: Amazon Nova Pro (best performance in benchmarks)
python run_comparison.py --limit 20 --models bedrock-nova-pro

# Compare top models from each provider
python run_comparison.py --limit 20 --models bedrock-nova-pro gpt4o claude gemini
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

### AWS Bedrock Models

| Model Key | Full Name | Model ID | Best For | Speed | Cost | Accuracy* |
|-----------|-----------|----------|----------|-------|------|-----------|
| `bedrock-nova-pro` 🏆 | **Amazon Nova Pro** | us.amazon.nova-pro-v1:0 | **Best overall** | ⚡⚡⚡ | $ | **60-82%** |
| `bedrock-nova-premier` | Amazon Nova Premier | us.amazon.nova-premier-v1:0 | Max capability | ⚡⚡ | $$ | TBD |
| `bedrock-claude` | Claude 4.5 Sonnet | us.anthropic.claude-sonnet-4-5-* | Via Bedrock | ⚡⚡⚡ | $$ | TBD |
| `bedrock-opus` | Claude 4.1 Opus | us.anthropic.claude-opus-4-1-* | Via Bedrock | ⚡ | $$$ | TBD |
| `bedrock-llama-3.3-70b` | Llama 3.3 70B | us.meta.llama3-3-70b-* | Open source | ⚡⚡ | $ | TBD |

**Setup**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` in .env
*Accuracy on ANLI contradiction detection benchmark (50 examples)

## Usage Examples

### Single Model Test
```bash
# Test Amazon Nova Pro (recommended)
python run_comparison.py --limit 20 --models bedrock-nova-pro

# Or test Gemini
python run_comparison.py --limit 20 --models gemini
```

### Head-to-Head Comparison
```bash
# Nova Pro vs GPT-4o
python run_comparison.py --limit 20 --models bedrock-nova-pro gpt4o

# Nova Pro vs Claude
python run_comparison.py --limit 20 --models bedrock-nova-pro claude

# All top models
python run_comparison.py --limit 20 --models bedrock-nova-pro gpt4o claude gemini
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

## Latest Benchmark Results (50 examples)

Based on standardized contradiction detection benchmarks:

### ANLI (Adversarial NLI)
| Model | Accuracy | F1 Score | Speed (s/example) | Winner |
|-------|----------|----------|-------------------|--------|
| Amazon Nova Pro | 0.60 | 0.52 | 13.9 | 🏆 **Best overall** |
| GPT-4 | 0.55 | 0.47 | 101* | - |
| Claude 3.7 Sonnet | 0.48 | 0.46 | 27.1 | - |
| Claude Sonnet | 0.45 | 0.27 | 1611* | - |

### SCITAIL (Scientific NLI)
| Model | Accuracy | Speed (s/example) | Winner |
|-------|----------|-------------------|--------|
| Amazon Nova Pro | 0.82 | 8.3 | 🏆 **Best** |
| Claude 3.7 Sonnet | 0.66 | 16.2 | - |

*Some earlier runs showed anomalous slow speeds possibly due to API issues.

**Key Insights:**
- Amazon Nova Pro delivers best accuracy across datasets
- Nova Pro is 2x faster than Claude 3.7 Sonnet
- Nova Pro provides excellent recall (73%) for catching contradictions

## Recommended Combinations

### Best Performance (Recommended)
```bash
# Amazon Nova Pro - proven best accuracy + speed
python run_comparison.py --limit 20 --models bedrock-nova-pro
```

### Quick Iteration
```bash
# Fast + cheap for rapid testing
python run_comparison.py --limit 10 --models gemini claude-haiku
```

### Quality Comparison
```bash
# Top performer vs other leading models
python run_comparison.py --limit 20 --models bedrock-nova-pro gpt4o claude gemini-pro
```

### Budget-Conscious
```bash
# Good results at low cost
python run_comparison.py --limit 50 --models bedrock-nova-pro gemini
```

### Comprehensive Analysis
```bash
# All top-tier models
python run_comparison.py --limit 20 --models bedrock-nova-pro gpt4 gpt4o claude claude-opus gemini-pro
```

## Adding Your API Keys

Edit `.env` file:

```bash
# AWS Bedrock (for Amazon Nova Pro - recommended)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

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

1. **Add API Keys**: Update `.env` with your keys (AWS recommended for Nova Pro)
2. **Quick Test**: `python run_comparison.py --limit 10 --models bedrock-nova-pro`
3. **Full Comparison**: `python run_comparison.py --limit 20 --models bedrock-nova-pro gpt4o claude gemini`
4. **Analyze**: Check results in `evaluation/reports/`

**Recommended Starting Point:** Amazon Nova Pro via AWS Bedrock has proven best performance in benchmarks (60-82% accuracy, 2x faster than alternatives).

Ready to run! 🚀
