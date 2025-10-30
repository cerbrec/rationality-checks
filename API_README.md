# Document Verification API

Flask-based REST API for running rationality checks on documents.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Make sure your `.env` file has the necessary API keys:

```bash
# For AWS Bedrock (recommended)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# OR for Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OR for OpenAI
OPENAI_API_KEY=sk-...

# OR for Gemini
GEMINI_API_KEY=...
```

### 3. Start the API Server

```bash
python api.py
```

By default, the server runs on `http://localhost:5000`

**Options:**
```bash
# Custom host/port
python api.py --host 0.0.0.0 --port 8080

# Debug mode
python api.py --debug
```

## API Endpoints

### GET `/`
API documentation and usage information

```bash
curl http://localhost:5000/
```

### GET `/health`
Health check endpoint

```bash
curl http://localhost:5000/health
```

### POST `/verify`
Verify a document for rationality and consistency

**Parameters (multipart/form-data):**
- `document` (file, required): Document to verify
- `provider` (string, optional): LLM provider - `anthropic`, `openai`, `bedrock`, `gemini` (default: `bedrock`)
- `model` (string, optional): Specific model to use
- `query` (string, optional): Context query for verification
- `verbose` (boolean, optional): Include all claims in response (default: `false`)

**Supported file formats:**
- Text files: `.md`, `.txt`, `.csv`, `.html` (all providers)
- Binary files: `.docx`, `.doc`, `.pdf`, `.xlsx`, `.xls` (bedrock only, max 4.5 MB)

## Usage Examples

### Using curl

**Basic verification:**
```bash
curl -X POST http://localhost:5000/verify \
  -F "document=@report.md" \
  -F "provider=bedrock"
```

**With verbose output:**
```bash
curl -X POST http://localhost:5000/verify \
  -F "document=@analysis.txt" \
  -F "provider=bedrock" \
  -F "verbose=true"
```

**With specific model:**
```bash
curl -X POST http://localhost:5000/verify \
  -F "document=@data.md" \
  -F "provider=bedrock" \
  -F "model=us.amazon.nova-pro-v1:0"
```

**Binary document (DOCX, PDF, etc.):**
```bash
curl -X POST http://localhost:5000/verify \
  -F "document=@report.docx" \
  -F "provider=bedrock"
```

### Using Python

**Simple client:**
```python
import requests

# Verify a document
with open('report.md', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/verify',
        files={'document': f},
        data={'provider': 'bedrock'}
    )

result = response.json()

# Check results
print(f"Total claims: {result['summary']['total_claims']}")
print(f"Accuracy: {result['summary']['accuracy_rate']}%")
print(f"Failed: {result['summary']['failed']}")

# Show failed claims
for claim in result['failed_claims']:
    print(f"\n❌ {claim['claim']}")
    for issue in claim['issues']:
        print(f"  - {issue}")
```

**Using the test client script:**
```bash
# Basic usage
python test_api_client.py report.md

# With options
python test_api_client.py document.txt --provider bedrock --verbose

# Binary document
python test_api_client.py analysis.docx --provider bedrock
```

### Using JavaScript/fetch

```javascript
const formData = new FormData();
formData.append('document', fileInput.files[0]);
formData.append('provider', 'bedrock');
formData.append('verbose', 'true');

fetch('http://localhost:5000/verify', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(result => {
    console.log('Accuracy:', result.summary.accuracy_rate + '%');
    console.log('Failed claims:', result.failed_claims.length);
  });
```

## Response Format

```json
{
  "success": true,
  "document": "report.md",
  "provider": "bedrock",
  "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  "summary": {
    "total_claims": 15,
    "passed": 12,
    "flagged": 2,
    "failed": 1,
    "accuracy_rate": 80.0
  },
  "failed_claims": [
    {
      "claim": "Company is valued at $50B with 10x revenue multiple and $7B revenue",
      "type": "quantitative",
      "confidence": 0.0,
      "recommendation": "revise",
      "issues": [
        "Constraint violated: 50B == 10 * 7B (50B ≠ 70B)"
      ]
    }
  ],
  "flagged_claims": [
    {
      "claim": "Has strong competitive moat",
      "type": "interpretive",
      "confidence": 0.7,
      "recommendation": "flag_uncertainty",
      "issues": [
        "Needs supporting evidence"
      ]
    }
  ],
  "improved_output": "...",
  "recommendation": {
    "level": "critical",
    "message": "Review and revise failed claims before publishing"
  }
}
```

## Error Responses

**400 Bad Request:**
```json
{
  "error": "No document file provided",
  "message": "Please upload a document file"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Verification failed",
  "message": "...",
  "type": "ValueError"
}
```

## Performance

- Text files: 2-3 minutes per verification
- Binary files (DOCX, PDF): 2-4 minutes per verification
- Max file size: 5 MB (4.5 MB for binary documents)
- Concurrent requests: Supported (each request runs independently)

## Development

### Run in debug mode
```bash
python api.py --debug
```

### Run on all interfaces
```bash
python api.py --host 0.0.0.0
```

### Custom port
```bash
python api.py --port 8080
```

## Integration with verify_document.py

The API wraps the functionality of `examples/verify_document.py` and provides the same capabilities:

| Feature | verify_document.py | API |
|---------|-------------------|-----|
| Text files | ✅ | ✅ |
| Binary files (Bedrock) | ✅ | ✅ |
| All providers | ✅ | ✅ |
| Custom models | ✅ | ✅ |
| Verbose output | ✅ | ✅ |
| JSON output | ✅ (file) | ✅ (response) |
| CLI interface | ✅ | ❌ |
| HTTP API | ❌ | ✅ |

## Troubleshooting

**"Could not connect to API server"**
- Make sure the API server is running: `python api.py`
- Check the host/port are correct

**"Provider initialization failed"**
- Check your `.env` file has the correct API keys
- Verify the provider name is correct (anthropic, openai, bedrock, gemini)

**"Unsupported file format"**
- Check the file extension is supported
- Binary files (.docx, .pdf, etc.) require `provider=bedrock`

**"File too large"**
- Maximum file size is 5 MB
- Binary documents have a 4.5 MB limit (Bedrock requirement)

## Security Notes

- API keys are loaded from `.env` file
- Never commit `.env` to version control
- Use HTTPS in production
- Consider adding authentication for production use
- Rate limiting recommended for production deployments

## Production Deployment

For production use, consider:

1. **WSGI server** (instead of Flask's dev server):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

2. **Reverse proxy** (nginx, Apache)
3. **HTTPS/SSL** certificates
4. **Authentication** middleware
5. **Rate limiting**
6. **Monitoring** and logging

## License

Same as parent project (MIT License)
