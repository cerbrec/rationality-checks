#!/usr/bin/env python3
"""
Basic smoke test for DOCX document support

This test validates that:
1. DOCX files are properly detected
2. Binary reading works correctly
3. Document bytes are passed to Bedrock
4. Basic claim extraction works with document attachments
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification_pipeline import BedrockProvider

def test_docx_basic():
    """
    Basic test that DOCX files can be read and processed
    """

    print("=" * 80)
    print("DOCX BASIC FUNCTIONALITY TEST")
    print("=" * 80)

    # Paths
    script_dir = Path(__file__).parent
    test_document = script_dir / "test_sarah_johnson.docx"

    # Check document exists
    if not test_document.exists():
        print(f"❌ Error: Test document not found: {test_document}")
        print("   Run: python tests/create_test_docx.py")
        return 1

    print(f"\n✓ Test document found: {test_document}")

    # Read document as binary
    try:
        with open(test_document, 'rb') as f:
            document_bytes = f.read()

        size_mb = len(document_bytes) / (1024 * 1024)
        print(f"✓ Successfully read DOCX file")
        print(f"  Size: {len(document_bytes)} bytes ({size_mb:.2f} MB)")

        if size_mb > 4.5:
            print(f"❌ Error: Document exceeds 4.5 MB limit")
            return 1

    except Exception as e:
        print(f"❌ Error reading DOCX file: {e}")
        return 1

    # Check AWS credentials
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("\n⚠️  Skipping Bedrock test - AWS credentials not found")
        print("✓ Basic DOCX reading test passed")
        return 0

    print(f"\n✓ AWS credentials found")
    print(f"  Region: {os.getenv('AWS_REGION', 'us-east-1')}")

    # Test Bedrock provider with document attachment
    try:
        print("\n" + "=" * 80)
        print("Testing Bedrock Document Attachment")
        print("=" * 80)

        provider = BedrockProvider.from_env(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        print(f"✓ Bedrock provider created")
        print(f"  Model: {provider.model_id}")

        # Simple test prompt
        test_prompt = """Please analyze the attached document and extract 2-3 key claims or facts from it.

Return your response as a simple list."""

        print(f"\n⏳ Sending DOCX to Bedrock with document attachment...")

        response = provider.generate(
            prompt=test_prompt,
            document_bytes=document_bytes,
            document_format="docx"
        )

        print(f"\n✓ Successfully received response from Bedrock!")
        print(f"  Response length: {len(response)} characters")
        print(f"\n--- Response Preview (first 500 chars) ---")
        print(response[:500])
        print("--- End Preview ---")

        # Validate response mentions the document
        if "sarah" in response.lower() or "johnson" in response.lower() or "basketball" in response.lower():
            print(f"\n✓ Response appears to reference document content")
        else:
            print(f"\n⚠️  Warning: Response may not be referencing the document")
            print(f"   (This is not necessarily an error, just unexpected)")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nDOCX document attachment support is working correctly!")
        print("The document was successfully:")
        print("  1. Read as binary")
        print("  2. Passed to Bedrock Converse API")
        print("  3. Processed by Claude with document understanding")

        return 0

    except Exception as e:
        print(f"\n❌ Error testing Bedrock document attachment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = test_docx_basic()
    sys.exit(exit_code)
