#!/usr/bin/env python3
"""
Test Bedrock Integration

Simple script to test AWS Bedrock connection and run verification.
Automatically loads .env file if present.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if AWS credentials are set
if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
    print("❌ Error: AWS credentials not found in environment")
    print("\nPlease either:")
    print("1. Create a .env file with:")
    print("   AWS_ACCESS_KEY_ID=your-key")
    print("   AWS_SECRET_ACCESS_KEY=your-secret")
    print("   AWS_REGION=us-east-1")
    print("\n2. Or set environment variables manually")
    sys.exit(1)

print("✓ AWS credentials loaded from environment")
print(f"  Region: {os.getenv('AWS_REGION', 'us-east-1')}")
print(f"  Access Key ID: {os.getenv('AWS_ACCESS_KEY_ID')[:10]}...")

# Import after loading env vars
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.verification_pipeline import BedrockProvider

# Test basic connection
print("\n" + "=" * 80)
print("Testing AWS Bedrock Connection")
print("=" * 80)

try:
    # Create provider
    print("\n[1/3] Creating Bedrock provider...")
    provider = BedrockProvider.from_env()
    print(f"✓ Provider created successfully")
    print(f"  Model: {provider.model_id}")

    # Test simple generation
    print("\n[2/3] Testing simple generation...")
    test_prompt = "Return only this exact JSON: {\"test\": \"success\"}"
    response = provider.generate(test_prompt)
    print(f"✓ Generation successful")
    print(f"  Response length: {len(response)} characters")
    print(f"  Response preview: {response[:200]}")

    # Test with system prompt
    print("\n[3/3] Testing with system prompt...")
    response2 = provider.generate(
        "What is 2+2?",
        system_prompt="You are a helpful math assistant. Be concise."
    )
    print(f"✓ System prompt test successful")
    print(f"  Response: {response2[:100]}")

    print("\n" + "=" * 80)
    print("✓ All Bedrock tests passed!")
    print("=" * 80)

    # Offer to run full verification test
    print("\nTo run full verification test with athlete report:")
    print("  python examples/athlete_report_test.py --provider=bedrock --auto")
    print("\nOr interactively (with pauses):")
    print("  python examples/athlete_report_test.py --provider=bedrock")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
