#!/usr/bin/env python3
"""
Download and prepare benchmark datasets for evaluation.

Datasets:
- ANLI: Adversarial Natural Language Inference (contradiction detection)
- FEVER: Fact Extraction and VERification (claim verification)
- LogiQA: Logical reasoning questions
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
import pandas as pd


class DatasetDownloader:
    """Download and prepare benchmark datasets"""

    def __init__(self, output_dir: str = "evaluation/datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_anli(self, sample_size: int = 100) -> Tuple[pd.DataFrame, Dict]:
        """
        Download ANLI dataset for contradiction detection testing.

        Args:
            sample_size: Number of examples to sample from each round

        Returns:
            DataFrame with samples and metadata dict
        """
        print(f"\n📥 Downloading ANLI dataset...")

        dataset = load_dataset("facebook/anli")

        # ANLI has 3 rounds (r1, r2, r3), each with train/dev/test
        # We'll use the test set from all rounds
        samples = []

        for round_name in ['test_r1', 'test_r2', 'test_r3']:
            round_data = dataset[round_name]

            # Sample examples
            num_samples = min(sample_size, len(round_data))
            indices = range(0, len(round_data), len(round_data) // num_samples)[:num_samples]

            for idx in indices:
                example = round_data[int(idx)]
                samples.append({
                    'id': f"anli_{round_name}_{example['uid']}",
                    'premise': example['premise'],
                    'hypothesis': example['hypothesis'],
                    'label': example['label'],  # 0=entailment, 1=neutral, 2=contradiction
                    'label_name': ['entailment', 'neutral', 'contradiction'][example['label']],
                    'round': round_name,
                    'dataset': 'anli'
                })

        df = pd.DataFrame(samples)

        # Save to JSON
        output_path = self.output_dir / 'anli_samples.json'
        df.to_json(output_path, orient='records', indent=2)
        print(f"✅ Saved {len(df)} ANLI samples to {output_path}")

        # Metadata
        metadata = {
            'dataset': 'ANLI',
            'total_samples': len(df),
            'label_distribution': df['label_name'].value_counts().to_dict(),
            'rounds': df['round'].unique().tolist(),
            'description': 'Adversarial Natural Language Inference for contradiction detection'
        }

        return df, metadata

    def download_fever(self, sample_size: int = 100) -> Tuple[pd.DataFrame, Dict]:
        """
        Download FEVER dataset for fact verification testing.

        Args:
            sample_size: Number of examples to sample

        Returns:
            DataFrame with samples and metadata dict
        """
        print(f"\n📥 Downloading FEVER dataset...")

        try:
            # FEVER v1.0 dataset
            dataset = load_dataset("fever", "v1.0", split='paper_test')

            # Sample examples
            num_samples = min(sample_size, len(dataset))
            indices = range(0, len(dataset), len(dataset) // num_samples)[:num_samples]

            samples = []
            for idx in indices:
                example = dataset[int(idx)]
                samples.append({
                    'id': f"fever_{example['id']}",
                    'claim': example['claim'],
                    'label': example['label'],  # SUPPORTS, REFUTES, NOT ENOUGH INFO
                    'evidence': example.get('evidence', ''),
                    'dataset': 'fever'
                })

            df = pd.DataFrame(samples)

            # Save to JSON
            output_path = self.output_dir / 'fever_samples.json'
            df.to_json(output_path, orient='records', indent=2)
            print(f"✅ Saved {len(df)} FEVER samples to {output_path}")

            # Metadata
            metadata = {
                'dataset': 'FEVER',
                'total_samples': len(df),
                'label_distribution': df['label'].value_counts().to_dict(),
                'description': 'Fact Extraction and VERification dataset'
            }

            return df, metadata

        except Exception as e:
            print(f"⚠️  FEVER download failed: {e}")
            print("   This is expected - FEVER may require manual download")
            return pd.DataFrame(), {'error': str(e)}

    def download_logiqa(self, sample_size: int = 100) -> Tuple[pd.DataFrame, Dict]:
        """
        Download LogiQA dataset for logical reasoning testing.

        Args:
            sample_size: Number of examples to sample

        Returns:
            DataFrame with samples and metadata dict
        """
        print(f"\n📥 Downloading LogiQA dataset...")

        try:
            # Try LogiQA 2.0 first
            dataset = load_dataset("lucasmccabe/logiqa", split='test')

            # Sample examples
            num_samples = min(sample_size, len(dataset))
            indices = range(0, len(dataset), len(dataset) // num_samples)[:num_samples]

            samples = []
            for idx in indices:
                example = dataset[int(idx)]
                samples.append({
                    'id': f"logiqa_{idx}",
                    'context': example.get('context', ''),
                    'question': example.get('question', ''),
                    'options': example.get('options', []),
                    'correct_option': example.get('correct_option', -1),
                    'dataset': 'logiqa'
                })

            df = pd.DataFrame(samples)

            # Save to JSON
            output_path = self.output_dir / 'logiqa_samples.json'
            df.to_json(output_path, orient='records', indent=2)
            print(f"✅ Saved {len(df)} LogiQA samples to {output_path}")

            # Metadata
            metadata = {
                'dataset': 'LogiQA',
                'total_samples': len(df),
                'description': 'Logical reasoning questions from civil service exams'
            }

            return df, metadata

        except Exception as e:
            print(f"⚠️  LogiQA download failed: {e}")
            print("   This is expected - LogiQA may require different loading")
            return pd.DataFrame(), {'error': str(e)}

    def download_all(self, sample_size: int = 100) -> Dict:
        """Download all datasets and return summary"""
        print("=" * 80)
        print("DOWNLOADING BENCHMARK DATASETS")
        print("=" * 80)

        results = {}

        # ANLI (most important for contradiction detection)
        anli_df, anli_meta = self.download_anli(sample_size)
        results['anli'] = {'data': anli_df, 'metadata': anli_meta}

        # FEVER (fact verification)
        fever_df, fever_meta = self.download_fever(sample_size)
        results['fever'] = {'data': fever_df, 'metadata': fever_meta}

        # LogiQA (logical reasoning)
        logiqa_df, logiqa_meta = self.download_logiqa(sample_size)
        results['logiqa'] = {'data': logiqa_df, 'metadata': logiqa_meta}

        # Save summary metadata
        summary = {
            'anli': anli_meta,
            'fever': fever_meta,
            'logiqa': logiqa_meta
        }

        summary_path = self.output_dir / 'datasets_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("DOWNLOAD SUMMARY")
        print("=" * 80)
        for dataset_name, data in results.items():
            meta = data['metadata']
            if 'error' in meta:
                print(f"\n❌ {dataset_name.upper()}: Failed")
                print(f"   Error: {meta['error']}")
            else:
                print(f"\n✅ {dataset_name.upper()}: {meta['total_samples']} samples")
                if 'label_distribution' in meta:
                    print(f"   Labels: {meta['label_distribution']}")

        print(f"\n📊 Summary saved to {summary_path}")

        return results


def main():
    """Main function to download datasets"""
    downloader = DatasetDownloader()

    # Download 100 samples from each dataset
    results = downloader.download_all(sample_size=100)

    # Print ANLI sample
    if not results['anli']['data'].empty:
        print("\n" + "=" * 80)
        print("SAMPLE ANLI EXAMPLE (Contradiction Detection)")
        print("=" * 80)
        sample = results['anli']['data'].iloc[0]
        print(f"ID: {sample['id']}")
        print(f"Premise: {sample['premise']}")
        print(f"Hypothesis: {sample['hypothesis']}")
        print(f"Label: {sample['label_name']}")


if __name__ == "__main__":
    main()
