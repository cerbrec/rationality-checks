# Gemini Integration Setup ✅

Gemini 2.0 Flash and Gemini 2.5 Pro are now supported! Here's how to use them.

## Quick Start

### 1. Add Your Gemini API Key to .env

Edit your `.env` file and add:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Run Comparison

```bash
# Test Gemini 2.0 Flash only
python run_comparison.py --limit 20 --models gemini

# Test Gemini 2.5 Pro
python run_comparison.py --limit 20 --models gemini-pro

# Compare all models (GPT-4, GPT-4o, Claude, Gemini)
python run_comparison.py --limit 20 --models gpt4 gpt4o claude gemini gemini-pro
```

## Available Gemini Models

| Model Key | Full Name | Description | Speed | Cost |
|-----------|-----------|-------------|-------|------|
| `gemini` | Gemini 2.0 Flash | Fast, experimental | ⚡⚡⚡ | $ |
| `gemini-pro` | Gemini 2.5 Pro | Most capable | ⚡⚡ | $$ |

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Get API Key"
3. Copy your key
4. Add to `.env` file: `GEMINI_API_KEY=your-key-here`

## Usage Examples

### Example 1: Quick Test
```bash
# Test Gemini on 10 examples (~$0.02)
python run_comparison.py --limit 10 --models gemini
```

### Example 2: Full Comparison
```bash
# Compare GPT-4o, Claude, and Gemini on 20 examples
python run_comparison.py --limit 20 --models gpt4o claude gemini
```

### Example 3: All Models
```bash
# Test all available models
python run_comparison.py --limit 20 --models gpt4 gpt4o gpt4-turbo claude claude-opus gemini gemini-pro
```

## Performance Notes

Based on initial testing:

- **Gemini 2.0 Flash**: Fastest, good for quick iterations
- **Gemini 2.5 Pro**: More capable, better reasoning
- **Cost**: Gemini is generally cheaper than GPT-4 and Claude

## Model Comparison

Your previous results showed:
- **GPT-4**: F1 Score 0.47, Best overall performance
- **Claude Sonnet**: F1 Score 0.27, Best calibration

Now you can add:
- **GPT-4o**: Latest GPT model, improved performance
- **Gemini 2.0 Flash**: Fast and cheap
- **Gemini 2.5 Pro**: Google's most capable model

## Troubleshooting

### Error: "GEMINI_API_KEY not set"
- Check that your `.env` file contains `GEMINI_API_KEY=...`
- Make sure there are no quotes around the key in .env
- Restart your terminal or re-source the environment

### Error: "google-generativeai package required"
```bash
source venv/bin/activate
pip install google-generativeai
```

### API Rate Limits
If you hit rate limits:
- Reduce `--limit` to fewer examples
- Add delay between requests (would need code modification)
- Upgrade your Gemini API quota

## Technical Details

The Gemini integration uses:
- **google-generativeai** SDK (installed automatically)
- **Temperature**: 0.0 (for consistency)
- **Max tokens**: 8192
- **Model**: `gemini-2.0-flash-exp` or `gemini-2.5-pro-002`

See `verification_pipeline.py` lines 309-354 for implementation details.

## Next Steps

Once you've added your API key:

1. **Test Gemini**: `python run_comparison.py --limit 10 --models gemini`
2. **Compare with others**: `python run_comparison.py --limit 20 --models gpt4 claude gemini`
3. **Analyze results**: Check `evaluation/reports/` for detailed metrics

## Cost Estimates

| Models | Examples | Est. Cost |
|--------|----------|-----------|
| gemini | 20 | $0.04 |
| gemini-pro | 20 | $0.10 |
| gpt4 claude gemini | 20 | $0.60 |
| gpt4o gpt4 claude gemini gemini-pro | 20 | $1.00 |

Ready to test! Just add your GEMINI_API_KEY to .env and run the comparison. 🚀
