#!/usr/bin/env python3
"""
Flask API for Document Verification

Provides HTTP endpoints for running rationality checks on documents.

Endpoints:
  POST /verify - Verify a document
  GET /health  - Health check

Usage:
  python api.py

  Or with custom host/port:
  python api.py --host 0.0.0.0 --port 8080
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integrated_verification import IntegratedVerificationPipeline
from src.verification_pipeline import (
    AnthropicProvider,
    OpenAIProvider,
    BedrockProvider,
    GeminiProvider
)

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Supported file extensions
TEXT_FORMATS = {'.md', '.txt', '.csv', '.html', '.htm'}
BINARY_FORMATS = {
    '.docx': 'docx',
    '.doc': 'doc',
    '.pdf': 'pdf',
    '.xlsx': 'xlsx',
    '.xls': 'xls'
}
ALL_FORMATS = TEXT_FORMATS | set(BINARY_FORMATS.keys())


def get_provider(provider_name: str, model: str = None):
    """Initialize the specified LLM provider"""

    if provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        model = model or "claude-3-5-sonnet-20241022"
        return AnthropicProvider(api_key=api_key, model=model)

    elif provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        model = model or "gpt-4o"
        return OpenAIProvider(api_key=api_key, model=model)

    elif provider_name == "bedrock":
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            raise ValueError("AWS_ACCESS_KEY_ID not found in environment")
        model = model or "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        return BedrockProvider(region_name=aws_region, model_id=model)

    elif provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        model = model or "gemini-2.0-flash-exp"
        return GeminiProvider(api_key=api_key, model=model)

    else:
        raise ValueError(f"Unknown provider '{provider_name}'. Available: anthropic, openai, bedrock, gemini")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'document-verification-api',
        'version': '1.0.0'
    })


@app.route('/verify', methods=['POST'])
def verify_document():
    """
    Verify a document for rationality and consistency

    Form data:
      - document (file, required): Document to verify
      - provider (string, optional): LLM provider (anthropic, openai, bedrock, gemini). Default: bedrock
      - model (string, optional): Specific model to use
      - query (string, optional): Context query for verification
      - verbose (boolean, optional): Include all claims in response. Default: false

    Returns:
      JSON with verification results
    """
    try:
        # Check if file is present
        if 'document' not in request.files:
            return jsonify({
                'error': 'No document file provided',
                'message': 'Please upload a document file'
            }), 400

        file = request.files['document']

        if file.filename == '':
            return jsonify({
                'error': 'Empty filename',
                'message': 'No file selected'
            }), 400

        # Get parameters
        provider_name = request.form.get('provider', 'bedrock')
        model = request.form.get('model', None)
        query = request.form.get('query', 'Analyze and verify the claims in this document')
        verbose = request.form.get('verbose', 'false').lower() == 'true'

        # Validate file extension
        filename = secure_filename(file.filename)
        file_ext = Path(filename).suffix.lower()

        if file_ext not in ALL_FORMATS:
            return jsonify({
                'error': 'Unsupported file format',
                'message': f'File extension {file_ext} not supported',
                'supported_formats': {
                    'text': list(TEXT_FORMATS),
                    'binary': list(BINARY_FORMATS.keys()) + ['(bedrock only)']
                }
            }), 400

        # Read file content
        document_text = None
        document_bytes = None
        document_format = None

        if file_ext in BINARY_FORMATS:
            # Binary document
            document_bytes = file.read()
            document_format = BINARY_FORMATS[file_ext]

            # Check size
            size_mb = len(document_bytes) / (1024 * 1024)
            if size_mb > 4.5:
                return jsonify({
                    'error': 'File too large',
                    'message': f'Document size ({size_mb:.2f} MB) exceeds 4.5 MB limit'
                }), 400

            # Check provider
            if provider_name != 'bedrock':
                return jsonify({
                    'error': 'Unsupported provider for binary documents',
                    'message': f'{document_format.upper()} files only supported with bedrock provider',
                    'current_provider': provider_name
                }), 400
        else:
            # Text document
            document_text = file.read().decode('utf-8')

        # Initialize provider
        try:
            llm = get_provider(provider_name, model)
        except ValueError as e:
            return jsonify({
                'error': 'Provider initialization failed',
                'message': str(e)
            }), 400

        # Create pipeline and run verification
        pipeline = IntegratedVerificationPipeline(llm)

        # Run verification
        if document_bytes is not None:
            report = pipeline.verify_analysis(
                original_output="",
                original_query=query,
                document_bytes=document_bytes,
                document_format=document_format
            )
        else:
            report = pipeline.verify_analysis(document_text, query)

        # Categorize results
        passed = [a for a in report.assessments if a.recommendation == "keep"]
        flagged = [a for a in report.assessments if a.recommendation in ["flag_uncertainty", "flag"]]
        failed = [a for a in report.assessments if a.recommendation in ["revise", "remove"]]

        # Calculate accuracy
        accuracy_rate = (len(passed) / len(report.assessments) * 100) if report.assessments else 0

        # Build response
        response = {
            'success': True,
            'document': filename,
            'provider': provider_name,
            'model': model,
            'summary': {
                'total_claims': len(report.assessments),
                'passed': len(passed),
                'flagged': len(flagged),
                'failed': len(failed),
                'accuracy_rate': round(accuracy_rate, 1)
            },
            'failed_claims': [
                {
                    'claim': a.claim.text,
                    'type': a.claim.claim_type.value,
                    'confidence': round(a.overall_confidence, 2),
                    'recommendation': a.recommendation,
                    'issues': [
                        issue
                        for result in a.verification_results
                        for issue in result.issues_found
                    ]
                }
                for a in failed
            ],
            'flagged_claims': [
                {
                    'claim': a.claim.text,
                    'type': a.claim.claim_type.value,
                    'confidence': round(a.overall_confidence, 2),
                    'recommendation': a.recommendation,
                    'issues': [
                        issue
                        for result in a.verification_results
                        for issue in result.issues_found
                    ]
                }
                for a in flagged
            ]
        }

        # Include all claims if verbose
        if verbose:
            response['all_claims'] = [
                {
                    'claim': a.claim.text,
                    'type': a.claim.claim_type.value,
                    'confidence': round(a.overall_confidence, 2),
                    'recommendation': a.recommendation,
                    'issues': [
                        issue
                        for result in a.verification_results
                        for issue in result.issues_found
                    ]
                }
                for a in report.assessments
            ]

        # Include improved output
        response['improved_output'] = report.improved_output

        # Include recommendations
        if failed:
            response['recommendation'] = {
                'level': 'critical',
                'message': 'Review and revise failed claims before publishing'
            }
        elif flagged:
            response['recommendation'] = {
                'level': 'warning',
                'message': 'Add supporting evidence for flagged claims'
            }
        else:
            response['recommendation'] = {
                'level': 'success',
                'message': 'Document appears sound and well-supported'
            }

        return jsonify(response), 200

    except Exception as e:
        # Log full traceback to console
        print("=" * 80)
        print("ERROR during verification:")
        traceback.print_exc()
        print("=" * 80)

        return jsonify({
            'error': 'Verification failed',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        'name': 'Document Verification API',
        'version': '1.0.0',
        'description': 'API for running rationality checks on documents',
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'POST /verify': 'Verify a document'
        },
        'verify_endpoint': {
            'method': 'POST',
            'content_type': 'multipart/form-data',
            'parameters': {
                'document': {
                    'type': 'file',
                    'required': True,
                    'description': 'Document to verify'
                },
                'provider': {
                    'type': 'string',
                    'required': False,
                    'default': 'bedrock',
                    'options': ['anthropic', 'openai', 'bedrock', 'gemini']
                },
                'model': {
                    'type': 'string',
                    'required': False,
                    'description': 'Specific model to use (provider-dependent)'
                },
                'query': {
                    'type': 'string',
                    'required': False,
                    'default': 'Analyze and verify the claims in this document'
                },
                'verbose': {
                    'type': 'boolean',
                    'required': False,
                    'default': False,
                    'description': 'Include all claims in response'
                }
            },
            'supported_formats': {
                'text': ['.md', '.txt', '.csv', '.html'],
                'binary': ['.docx', '.doc', '.pdf', '.xlsx', '.xls (bedrock only)']
            }
        },
        'examples': {
            'curl': '''
curl -X POST http://localhost:5000/verify \\
  -F "document=@report.md" \\
  -F "provider=bedrock" \\
  -F "verbose=true"
            '''.strip(),
            'python': '''
import requests

with open('report.md', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/verify',
        files={'document': f},
        data={
            'provider': 'bedrock',
            'verbose': 'true'
        }
    )

result = response.json()
print(f"Accuracy: {result['summary']['accuracy_rate']}%")
            '''.strip()
        }
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Document Verification API server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print("=" * 80)
    print("Document Verification API")
    print("=" * 80)
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print(f"  GET  http://{args.host}:{args.port}/         - API documentation")
    print(f"  GET  http://{args.host}:{args.port}/health   - Health check")
    print(f"  POST http://{args.host}:{args.port}/verify   - Verify document")
    print("\nExample usage:")
    print(f'''  curl -X POST http://{args.host}:{args.port}/verify \\
    -F "document=@report.md" \\
    -F "provider=bedrock"''')
    print("\n" + "=" * 80)

    app.run(host=args.host, port=args.port, debug=args.debug)
