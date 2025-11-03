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
import requests
import logging
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from urllib.parse import urlparse
from supabase import create_client, Client
import httpx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Use service key for backend operations

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("⚠️  SUPABASE_URL or SUPABASE_SERVICE_KEY not configured. Job queue will not work.")
    supabase: Client = None
else:
    # Log Supabase URL for debugging (mask for security)
    masked_url = SUPABASE_URL[:8] + "..." + SUPABASE_URL[-10:] if len(SUPABASE_URL) > 20 else "***"
    logger.info(f"🔧 Connecting to Supabase: {masked_url}")
    try:
        # Configure httpx client with longer timeout and connection settings
        httpx_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            follow_redirects=True
        )

        supabase: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            options={
                'postgrest_client_timeout': 30,
                'storage_client_timeout': 30,
            }
        )
        logger.info("✅ Supabase client initialized successfully")

        # Test connection
        try:
            logger.info("🧪 Testing Supabase connection...")
            test_response = supabase.table('verification_jobs').select('id').limit(1).execute()
            logger.info("✅ Supabase connection test successful")
        except Exception as test_error:
            logger.error(f"⚠️  Supabase connection test failed: {test_error}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase client: {e}")
        supabase: Client = None

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

# Configure CORS
CORS(app, origins=[
    'https://verque-frontend.vercel.app',  # Production Vercel frontend
    'http://localhost:3000',                # Local development (React/Next.js)
    'http://localhost:5173',                # Local development (Vite)
    'http://localhost:8080',                # Local development (alternative)
],
supports_credentials=True,
allow_headers=['Content-Type', 'Authorization'],
methods=['GET', 'POST', 'OPTIONS'])

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

class JobStatus:
    """Job status constants"""
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'


def create_job(job_data):
    """Create a new job in Supabase and return job ID"""
    if not supabase:
        raise ValueError("Supabase not configured")

    # Prepare job record
    job_record = {
        'status': JobStatus.PENDING,
        'progress': 0,
        'document_text': job_data.get('document_text'),
        'document_url': job_data.get('document_url'),
        'document_format': job_data.get('document_format'),
        'filename': job_data.get('filename'),
        'provider_name': job_data.get('provider_name', 'bedrock'),
        'model': job_data.get('model'),
        'query': job_data.get('query', 'Analyze and verify the claims in this document'),
        'is_verbose': job_data.get('verbose', False),  # Map 'verbose' to 'is_verbose' for database
        'claim_id': job_data.get('claim_id')
    }

    # Note: document_bytes cannot be stored directly in JSON
    # For binary documents, we'll need to handle them differently
    # Either: 1) store in Supabase Storage and reference URL, or 2) base64 encode
    if job_data.get('document_bytes'):
        # For now, we'll skip binary document support in async mode
        # This can be enhanced later with Supabase Storage
        raise ValueError("Binary document uploads not yet supported in async mode. Please use text or URL input.")

    # Insert into Supabase
    try:
        logger.info(f"📤 Inserting job into Supabase...")
        response = supabase.table('verification_jobs').insert(job_record).execute()

        if not response.data:
            raise ValueError("Failed to create job in Supabase")

        job_id = response.data[0]['id']
        logger.info(f"✅ Created job {job_id} in Supabase")
        return job_id
    except Exception as e:
        logger.error(f"❌ Supabase insert failed: {type(e).__name__}: {e}")
        logger.error(f"🔍 SUPABASE_URL being used: {SUPABASE_URL[:30]}...")
        raise


def update_job(job_id, status=None, result=None, error=None, progress=None):
    """Update job status in Supabase"""
    if not supabase:
        return

    update_data = {}

    if status:
        update_data['status'] = status
        if status == JobStatus.PROCESSING and 'started_at' not in update_data:
            update_data['started_at'] = datetime.utcnow().isoformat()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            update_data['completed_at'] = datetime.utcnow().isoformat()

    if result is not None:
        update_data['result'] = result

    if error:
        update_data['error'] = error

    if progress is not None:
        update_data['progress'] = progress

    if update_data:
        try:
            supabase.table('verification_jobs').update(update_data).eq('id', job_id).execute()
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")


def get_job(job_id):
    """Get job details from Supabase"""
    if not supabase:
        return None

    try:
        response = supabase.table('verification_jobs').select('*').eq('id', job_id).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}")
        return None


def process_verification_job(job_id):
    """Background worker to process verification job"""
    try:
        job = get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        update_job(job_id, status=JobStatus.PROCESSING, progress=10)
        logger.info(f"🔄 Processing job {job_id}")

        # Get job data directly from job record
        document_text = job.get('document_text')
        document_url = job.get('document_url')
        document_format = job.get('document_format')
        filename = job.get('filename', 'unknown')
        provider_name = job.get('provider_name', 'bedrock')
        model = job.get('model')
        query = job.get('query', 'Analyze and verify the claims in this document')
        verbose = job.get('is_verbose', False)  # Database uses 'is_verbose'

        # Fetch URL if provided
        if document_url and not document_text:
            logger.info(f"🌐 Fetching content from URL: {document_url}")
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Cache-Control': 'max-age=0'
                }
                response = requests.get(document_url, headers=headers, timeout=30, allow_redirects=True)
                response.raise_for_status()
                document_text = response.text
                logger.info(f"✅ Fetched {len(document_text)} characters from URL")
            except Exception as e:
                logger.error(f"Failed to fetch URL: {e}")
                update_job(job_id, status=JobStatus.FAILED, error={'message': f'Failed to fetch URL: {str(e)}'})
                return

        # Initialize provider
        update_job(job_id, progress=20)
        logger.info(f"🤖 Initializing LLM provider: {provider_name}" + (f" (model: {model})" if model else ""))
        llm = get_provider(provider_name, model)

        # Create pipeline
        update_job(job_id, progress=30)
        logger.info("🔧 Creating verification pipeline...")
        pipeline = IntegratedVerificationPipeline(llm)

        # Run verification (text only for async jobs)
        update_job(job_id, progress=40)
        logger.info("🔍 Starting verification process...")
        logger.info(f"   → Processing text content ({len(document_text)} characters)")
        report = pipeline.verify_analysis(document_text, query)

        update_job(job_id, progress=90)
        logger.info("✅ Verification complete!")

        # Categorize results
        passed = [a for a in report.assessments if a.recommendation == "keep"]
        flagged = [a for a in report.assessments if a.recommendation in ["flag_uncertainty", "flag"]]
        failed = [a for a in report.assessments if a.recommendation in ["revise", "remove"]]

        # Calculate accuracy
        accuracy_rate = (len(passed) / len(report.assessments) * 100) if report.assessments else 0

        # Log results summary
        logger.info("=" * 80)
        logger.info("📊 VERIFICATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total claims analyzed: {len(report.assessments)}")
        logger.info(f"  ✅ Passed: {len(passed)}")
        logger.info(f"  ⚠️  Flagged: {len(flagged)}")
        logger.info(f"  ❌ Failed: {len(failed)}")
        logger.info(f"Accuracy Rate: {accuracy_rate:.1f}%")
        logger.info("=" * 80)

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

        update_job(job_id, status=JobStatus.COMPLETED, result=response, progress=100)
        logger.info(f"✅ Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        traceback.print_exc()
        update_job(
            job_id,
            status=JobStatus.FAILED,
            error={
                'message': str(e),
                'type': type(e).__name__
            }
        )


def poll_and_process_jobs():
    """Background thread that continuously polls for pending jobs"""
    logger.info("🔄 Starting job polling worker...")

    while True:
        try:
            if not supabase:
                time.sleep(10)
                continue

            # Query for pending jobs (ordered by creation time)
            response = supabase.table('verification_jobs')\
                .select('id')\
                .eq('status', JobStatus.PENDING)\
                .order('created_at')\
                .limit(1)\
                .execute()

            if response.data and len(response.data) > 0:
                job_id = response.data[0]['id']
                logger.info(f"📋 Found pending job: {job_id}")

                # Process the job in this thread
                process_verification_job(job_id)

            # Poll every 5 seconds
            time.sleep(5)

        except Exception as e:
            logger.error(f"Error in job polling worker: {e}")
            time.sleep(10)


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

    Accepts either:
      1. Multipart form data with file upload
      2. JSON body with raw text content
      3. JSON body with URL to fetch content from

    Form data (multipart/form-data):
      - document (file, required): Document to verify
      - provider (string, optional): LLM provider (anthropic, openai, bedrock, gemini). Default: bedrock
      - model (string, optional): Specific model to use
      - query (string, optional): Context query for verification
      - verbose (boolean, optional): Include all claims in response. Default: false

    JSON body (application/json):
      - text (string, required if url not provided): Raw document text to verify
      - url (string, required if text not provided): URL to fetch document from
      - provider (string, optional): LLM provider. Default: bedrock
      - model (string, optional): Specific model to use
      - query (string, optional): Context query for verification
      - verbose (boolean, optional): Include all claims in response. Default: false

    Returns:
      JSON with verification results
    """
    try:
        document_text = None
        document_bytes = None
        document_format = None
        filename = None

        # Check if this is a JSON request with raw text or URL
        if request.is_json:
            logger.info("📥 Received JSON request")
            data = request.get_json()

            # Check if either 'text' or 'url' is provided
            if 'text' not in data and 'url' not in data:
                return jsonify({
                    'error': 'No content provided',
                    'message': 'JSON body must include either "text" field with document content or "url" field with URL to fetch'
                }), 400

            if 'text' in data and 'url' in data:
                return jsonify({
                    'error': 'Multiple content sources',
                    'message': 'Provide either "text" or "url", not both'
                }), 400

            # Get parameters from JSON
            provider_name = data.get('provider', 'bedrock')
            model = data.get('model', None)
            query = data.get('query', 'Analyze and verify the claims in this document')
            verbose = data.get('verbose', False)

            # Handle raw text
            if 'text' in data:
                document_text = data.get('text')

                if not document_text or not document_text.strip():
                    return jsonify({
                        'error': 'Empty text',
                        'message': 'Text content cannot be empty'
                    }), 400

                filename = 'raw_text_input'
                logger.info(f"📄 Processing raw text input: {len(document_text)} characters")

            # Handle URL
            elif 'url' in data:
                url = data.get('url')

                if not url or not url.strip():
                    return jsonify({
                        'error': 'Empty URL',
                        'message': 'URL cannot be empty'
                    }), 400

                # Validate URL format
                try:
                    parsed_url = urlparse(url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        return jsonify({
                            'error': 'Invalid URL',
                            'message': 'URL must be a valid HTTP/HTTPS URL'
                        }), 400
                except Exception:
                    return jsonify({
                        'error': 'Invalid URL',
                        'message': 'Could not parse URL'
                    }), 400

                # Fetch content from URL
                try:
                    logger.info(f"🌐 Fetching content from URL: {url}")

                    # Add browser-like headers to avoid 403 errors
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Cache-Control': 'max-age=0'
                    }

                    response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                    response.raise_for_status()

                    # Try to get text content
                    document_text = response.text

                    if not document_text or not document_text.strip():
                        return jsonify({
                            'error': 'Empty content',
                            'message': 'URL returned empty content'
                        }), 400

                    filename = f'url_fetch_{parsed_url.netloc}'
                    logger.info(f"✅ Successfully fetched {len(document_text)} characters from URL")

                except requests.exceptions.Timeout:
                    return jsonify({
                        'error': 'Request timeout',
                        'message': 'Request to URL timed out after 30 seconds'
                    }), 400
                except requests.exceptions.ConnectionError:
                    return jsonify({
                        'error': 'Connection error',
                        'message': 'Could not connect to URL'
                    }), 400
                except requests.exceptions.HTTPError as e:
                    return jsonify({
                        'error': 'HTTP error',
                        'message': f'URL returned HTTP {e.response.status_code}'
                    }), 400
                except Exception as e:
                    return jsonify({
                        'error': 'URL fetch failed',
                        'message': str(e)
                    }), 400

        # Otherwise handle as file upload
        elif 'document' in request.files:
            logger.info("📁 Received file upload")
            file = request.files['document']

            if file.filename == '':
                return jsonify({
                    'error': 'Empty filename',
                    'message': 'No file selected'
                }), 400

            # Get parameters from form
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
            if file_ext in BINARY_FORMATS:
                # Binary document
                document_bytes = file.read()
                document_format = BINARY_FORMATS[file_ext]

                # Check size
                size_mb = len(document_bytes) / (1024 * 1024)
                logger.info(f"📄 Processing binary document: {filename} ({document_format}, {size_mb:.2f} MB)")
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
                logger.info(f"📄 Processing text document: {filename} ({len(document_text)} characters)")

        else:
            return jsonify({
                'error': 'No document provided',
                'message': 'Please provide either a file upload (multipart/form-data) or raw text (JSON with "text" field)'
            }), 400

        # Validate provider before creating job
        try:
            get_provider(provider_name, model)
        except ValueError as e:
            logger.error(f"❌ Provider validation failed: {e}")
            return jsonify({
                'error': 'Provider initialization failed',
                'message': str(e)
            }), 400

        # Create job data
        job_data = {
            'document_text': document_text,
            'document_url': url if 'url' in locals() else None,  # Store URL if it was fetched
            'document_bytes': document_bytes,
            'document_format': document_format,
            'filename': filename,
            'provider_name': provider_name,
            'model': model,
            'query': query,
            'verbose': verbose
        }

        # Create job in Supabase (background worker will pick it up automatically)
        job_id = create_job(job_data)
        logger.info(f"✅ Created job {job_id}")

        # Return job ID immediately
        # The background polling worker will automatically pick up and process this job
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': JobStatus.PENDING,
            'message': 'Verification job created. Poll /verify/{job_id} for status and results.'
        }), 202

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


@app.route('/verify/<job_id>', methods=['GET'])
def get_verification_status(job_id):
    """
    Get the status of a verification job

    Returns:
      - status: pending, processing, completed, or failed
      - progress: 0-100
      - result: verification results (only when completed)
      - error: error details (only when failed)
    """
    job = get_job(job_id)

    if not job:
        return jsonify({
            'error': 'Job not found',
            'message': f'No job found with ID {job_id}'
        }), 404

    response = {
        'job_id': job['id'],
        'status': job['status'],
        'progress': job['progress'],
        'created_at': job['created_at'],
        'updated_at': job['updated_at']
    }

    if job['status'] == JobStatus.COMPLETED:
        response['result'] = job['result']
    elif job['status'] == JobStatus.FAILED:
        response['error'] = job['error']

    return jsonify(response), 200


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
            'content_types': ['multipart/form-data', 'application/json'],
            'description': 'Accepts either file uploads or raw text in JSON body',
            'file_upload_params': {
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
                }
            },
            'json_body_params': {
                'content_type': 'application/json',
                'description': 'Provide either "text" or "url", not both',
                'parameters': {
                    'text': {
                        'type': 'string',
                        'required': 'Either text or url required',
                        'description': 'Raw document text to verify'
                    },
                    'url': {
                        'type': 'string',
                        'required': 'Either text or url required',
                        'description': 'URL to fetch document content from'
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
                }
            },
            'supported_formats': {
                'text': ['.md', '.txt', '.csv', '.html'],
                'binary': ['.docx', '.doc', '.pdf', '.xlsx', '.xls (bedrock only)']
            }
        },
        'examples': {
            'file_upload_curl': '''
curl -X POST http://localhost:5000/verify \\
  -F "document=@report.md" \\
  -F "provider=bedrock" \\
  -F "verbose=true"
            '''.strip(),
            'raw_text_curl': '''
curl -X POST http://localhost:5000/verify \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Your document text here...",
    "provider": "bedrock",
    "verbose": true
  }'
            '''.strip(),
            'file_upload_python': '''
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
            '''.strip(),
            'raw_text_python': '''
import requests

response = requests.post(
    'http://localhost:5000/verify',
    json={
        'text': 'Your document text here...',
        'provider': 'bedrock',
        'verbose': True
    }
)

result = response.json()
print(f"Accuracy: {result['summary']['accuracy_rate']}%")
            '''.strip(),
            'url_fetch_curl': '''
curl -X POST http://localhost:5000/verify \\
  -H "Content-Type: application/json" \\
  -d '{
    "url": "https://example.com/document.txt",
    "provider": "bedrock",
    "verbose": true
  }'
            '''.strip(),
            'url_fetch_python': '''
import requests

response = requests.post(
    'http://localhost:5000/verify',
    json={
        'url': 'https://example.com/document.txt',
        'provider': 'bedrock',
        'verbose': True
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
    print(f"  POST http://{args.host}:{args.port}/verify   - Verify document (async)")
    print(f"  GET  http://{args.host}:{args.port}/verify/<job_id> - Check job status")
    print("\nExample usage:")
    print(f'''  curl -X POST http://{args.host}:{args.port}/verify \\
    -F "document=@report.md" \\
    -F "provider=bedrock"''')
    print("\n" + "=" * 80)

    # Start background job processing worker
    if supabase:
        worker_thread = threading.Thread(target=poll_and_process_jobs, daemon=True)
        worker_thread.start()
        logger.info("✅ Background job worker started")
    else:
        logger.warning("⚠️  Background job worker NOT started (Supabase not configured)")

    app.run(host=args.host, port=args.port, debug=args.debug)
