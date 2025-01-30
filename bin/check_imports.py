#!/usr/bin/env python3
"""Test imports for evaluation pipeline."""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path('/Users/mogen/Desktop/Research_Case')
sys.path.append(str(project_root))

def test_imports():
    """Test importing each component separately."""
    try:
        print("1. Importing rouge_evaluator...")
        from research_case.evaluator.rouge_evaluator import RougeEvaluator
        print("   Success!")
        
        print("\n2. Importing similarity_analyzer...")
        from research_case.evaluator.similarity_analyzer import SimilarityAnalyzer
        print("   Success!")
        
        print("\n3. Importing llm_judge...")
        from research_case.evaluator.llm_judge import LLMJudge
        print("   Success!")
        
        print("\n4. Importing pipeline...")
        from research_case.evaluator.pipeline import EvaluationPipeline
        print("   Success!")
        
        print("\nAll imports successful!")
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("\nCurrent directory structure:")
        os.system('ls -R research_case/evaluator/')
        
        print("\nPython path:")
        for path in sys.path:
            print(f"  {path}")
            
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()