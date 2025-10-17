#!/usr/bin/env python3
"""
Try alternative datasets when primary ones fail
"""

import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def try_vitaminc_factcheck(sample_size: int = 100):
    """
    VitaminC is a fact verification dataset similar to FEVER
    """
    print(f"\n📥 Trying VitaminC (FEVER alternative)...")
    try:
        # VitaminC: Fact Verification in Context
        dataset = load_dataset("tals/vitaminc", split='test')

        num_samples = min(sample_size, len(dataset))
        indices = range(0, len(dataset), len(dataset) // num_samples)[:num_samples]

        samples = []
        for idx in indices:
            example = dataset[int(idx)]
            samples.append({
                'id': f"vitaminc_{idx}",
                'claim': example['claim'],
                'evidence': example.get('evidence', ''),
                'label': example.get('label', ''),
                'dataset': 'vitaminc'
            })

        df = pd.DataFrame(samples)
        output_path = Path('evaluation/datasets/vitaminc_samples.json')
        df.to_json(output_path, orient='records', indent=2)
        print(f"✅ Saved {len(df)} VitaminC samples")
        return df
    except Exception as e:
        print(f"⚠️  VitaminC failed: {e}")
        return None


def try_scitail_nli(sample_size: int = 100):
    """
    SciTail is an NLI dataset with entailment/neutral labels
    """
    print(f"\n📥 Trying SciTail (NLI dataset)...")
    try:
        dataset = load_dataset("allenai/scitail", "snli_format", split='test')

        num_samples = min(sample_size, len(dataset))
        indices = range(0, len(dataset), len(dataset) // num_samples)[:num_samples]

        samples = []
        for idx in indices:
            example = dataset[int(idx)]
            samples.append({
                'id': f"scitail_{idx}",
                'premise': example['sentence1'],
                'hypothesis': example['sentence2'],
                'label': example['gold_label'],
                'dataset': 'scitail'
            })

        df = pd.DataFrame(samples)
        output_path = Path('evaluation/datasets/scitail_samples.json')
        df.to_json(output_path, orient='records', indent=2)
        print(f"✅ Saved {len(df)} SciTail samples")
        return df
    except Exception as e:
        print(f"⚠️  SciTail failed: {e}")
        return None


def try_snli(sample_size: int = 100):
    """
    SNLI is a classic NLI dataset
    """
    print(f"\n📥 Trying SNLI (Classic NLI)...")
    try:
        dataset = load_dataset("stanfordnlp/snli", split='test', trust_remote_code=True)

        # Filter out examples with label -1 (no consensus)
        dataset = dataset.filter(lambda x: x['label'] != -1)

        num_samples = min(sample_size, len(dataset))
        indices = range(0, len(dataset), len(dataset) // num_samples)[:num_samples]

        samples = []
        for idx in indices:
            example = dataset[int(idx)]
            samples.append({
                'id': f"snli_{idx}",
                'premise': example['premise'],
                'hypothesis': example['hypothesis'],
                'label': example['label'],  # 0=entailment, 1=neutral, 2=contradiction
                'label_name': ['entailment', 'neutral', 'contradiction'][example['label']],
                'dataset': 'snli'
            })

        df = pd.DataFrame(samples)
        output_path = Path('evaluation/datasets/snli_samples.json')
        df.to_json(output_path, orient='records', indent=2)
        print(f"✅ Saved {len(df)} SNLI samples")
        return df
    except Exception as e:
        print(f"⚠️  SNLI failed: {e}")
        return None


def try_logiqa_alternatives(sample_size: int = 100):
    """
    Try alternative logical reasoning datasets
    """
    print(f"\n📥 Trying ReClor (Logical reasoning from LSAT)...")
    try:
        dataset = load_dataset("yangdong/reclor", split='test')

        num_samples = min(sample_size, len(dataset))
        indices = range(0, len(dataset), len(dataset) // num_samples)[:num_samples]

        samples = []
        for idx in indices:
            example = dataset[int(idx)]
            samples.append({
                'id': f"reclor_{idx}",
                'context': example['context'],
                'question': example['question'],
                'answers': example['answers'],
                'correct_answer': example.get('label', -1),
                'dataset': 'reclor'
            })

        df = pd.DataFrame(samples)
        output_path = Path('evaluation/datasets/reclor_samples.json')
        df.to_json(output_path, orient='records', indent=2)
        print(f"✅ Saved {len(df)} ReClor samples")
        return df
    except Exception as e:
        print(f"⚠️  ReClor failed: {e}")
        return None


def main():
    """Try alternative datasets"""
    print("=" * 80)
    print("TRYING ALTERNATIVE DATASETS")
    print("=" * 80)

    results = {}

    # Try fact verification alternatives
    vitaminc = try_vitaminc_factcheck(100)
    if vitaminc is not None:
        results['vitaminc'] = vitaminc

    # Try NLI alternatives
    scitail = try_scitail_nli(100)
    if scitail is not None:
        results['scitail'] = scitail

    snli = try_snli(100)
    if snli is not None:
        results['snli'] = snli

    # Try logical reasoning alternatives
    reclor = try_logiqa_alternatives(100)
    if reclor is not None:
        results['reclor'] = reclor

    print("\n" + "=" * 80)
    print("ALTERNATIVE DATASETS SUMMARY")
    print("=" * 80)
    for name, df in results.items():
        print(f"✅ {name.upper()}: {len(df)} samples downloaded")

    return results


if __name__ == "__main__":
    main()
