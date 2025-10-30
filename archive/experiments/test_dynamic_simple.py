#!/usr/bin/env python3
"""
Simple test to verify dynamic claims discovery
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integrated_verification import IntegratedVerificationPipeline
from src.verification_pipeline import BedrockProvider

# Read document
doc_path = Path("~/Downloads/BYU_Football_NIL_Analysis.md").expanduser()
with open(doc_path) as f:
    document = f.read()

print("=== TESTING DYNAMIC CLAIMS DISCOVERY ===\n")

# Initialize provider
print("1. Initializing BedrockProvider...")
provider = BedrockProvider(
    region_name="us-east-1",
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
print(f"   Provider type: {type(provider)}\n")

# Initialize pipeline
print("2. Initializing pipeline...")
pipeline = IntegratedVerificationPipeline(provider)
print(f"   Pipeline LLM type: {type(pipeline.llm)}\n")

# Run verification with enable_dynamic_claims=True
print("3. Running verification with dynamic claims enabled...")
print("   (This will take 2-3 minutes)\n")

try:
    report = pipeline.verify_analysis(
        document,
        "Verify this BYU NIL analysis",
        enable_tool_use=True,
        enable_dynamic_claims=True
    )

    print("\n=== RESULTS ===")
    print(f"Total claims: {len(report.assessments)}")
    print(f"Passed: {len([a for a in report.assessments if a.recommendation == 'keep'])}")
    print(f"Failed: {len([a for a in report.assessments if a.recommendation in ['revise', 'remove']])}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
