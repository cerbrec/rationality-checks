# Pydantic Validation for JSON Responses

## Overview

The verification pipeline now uses **Pydantic v2** for robust JSON parsing and validation. This provides:

1. **Type Safety**: Automatic validation of field types
2. **Schema Validation**: Ensures all required fields are present
3. **Clear Error Messages**: Detailed feedback when validation fails
4. **Self-Documenting**: Pydantic models serve as schema documentation

## Pydantic Models

### ClaimExtractionSchema
Validates individual claim extraction with:
- `id`, `text`, `claim_type`, `source_section` (required)
- `dependencies`, `context` (optional)
- `is_formalizable` (boolean)
- `formal_structure` (optional nested structure)

**Validation Rules:**
- `claim_type` must be one of: `factual`, `quantitative`, `causal`, `logical`, `interpretive`, `predictive`, `assumption`
- Invalid claim types are rejected with clear error messages

### FormalStructureSchema
Validates formal structure with:
- `propositions`: List of `PropositionSchema` (subject, predicate, value)
- `constraints`: List of `ConstraintSchema` (variables, formula)
- `implications`: List of strings

### VerificationResultSchema
Validates verification results with:
- `claim_id`, `passed`, `confidence` (required)
- `confidence` must be between 0.0 and 1.0
- `issues_found`, `evidence`, `suggested_revision` (optional)

## Usage

### Before (Raw JSON parsing):
```python
response = self.llm.generate(prompt)
data = json.loads(response)  # No validation!
claims = [process(c) for c in data["claims"]]
```

### After (Pydantic validation):
```python
response = self.llm.generate(prompt)
parsed = ClaimExtractionResponse.parse_raw(response)  # Validated!
claims = [convert(c) for c in parsed.claims]
```

## Error Handling

The system now provides three levels of error reporting:

1. **JSON Parse Error**: Invalid JSON syntax
   ```
   ❌ Failed to parse JSON response
   Error: Expecting value: line 1 column 1 (char 0)
   ```

2. **Schema Validation Error**: Valid JSON but wrong structure
   ```
   ❌ Failed to validate response schema
   Error: claim_type must be one of {...}, got 'comparative'
   ```

3. **Successful Validation**: Proceeds with verified data
   ```
   ✓ Found 23 claims (18 formalizable, 5 interpretive)
   ```

## Benefits

### Type Safety
```python
# Pydantic ensures types are correct
claim.confidence  # Guaranteed to be float between 0.0-1.0
claim.evidence    # Guaranteed to be list of EvidenceSchema
```

### Clear Documentation
```python
class ClaimExtractionSchema(BaseModel):
    """Schema for a single extracted claim"""
    id: str
    text: str
    claim_type: str  # Must be one of valid ClaimType values
    # ... clearly documented fields
```

### Automatic Validation
```python
@validator('claim_type')
def validate_claim_type(cls, v):
    """Ensure claim_type is valid"""
    valid_types = {'factual', 'quantitative', ...}
    if v not in valid_types:
        raise ValueError(f"Invalid: {v}")
    return v
```

## Testing

Run tests with Pydantic validation:
```bash
# Test with mock provider (no API calls)
python examples/athlete_report_test.py --provider=mock --auto

# Test with Bedrock (full validation)
python examples/athlete_report_test.py --provider=bedrock --auto
```

## Dependencies

Added to `requirements.txt`:
```
pydantic>=2.10.0  # For data validation and JSON schema
```

Install with:
```bash
pip install pydantic>=2.10.0
```

## Future Enhancements

Potential improvements:
1. Add JSON Schema export for LLM prompts
2. Implement strict mode for additional validation
3. Add custom validators for domain-specific rules
4. Generate OpenAPI specs from Pydantic models
