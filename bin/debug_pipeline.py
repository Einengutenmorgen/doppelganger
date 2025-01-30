#!/usr/bin/env python3
"""Debug script for evaluation pipeline."""
import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check environment setup."""
    logger.debug("Checking environment variables...")
    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f".env file exists: {os.path.exists('.env')}")
    
    required_vars = ['OPENAI_API_KEY']
    env_status = {}
    
    for var in required_vars:
        value = os.getenv(var)
        env_status[var] = 'Present' if value else 'Missing'
        if value:
            logger.debug(f"{var}: Found (starts with {value[:4]}...)")
        else:
            logger.debug(f"{var}: Not found")
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return False
    return True

def check_file_structure():
    """Check if all required files and directories exist."""
    logger.debug("Checking file structure...")
    current_dir = Path.cwd()
    logger.debug(f"Current directory: {current_dir}")
    
    # Check required directories
    required_dirs = [
        'research_case/processors',
        'research_case/analyzers',
        'research_case/generators',
        'research_case/evaluator'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = current_dir / dir_path
        logger.debug(f"Checking directory: {full_path}")
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            logger.debug(f"Found directory: {dir_path}")
    
    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        return False
    
    # Check required files
    required_files = [
        'setup.py',
        'requirements.txt',
        '.env'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = current_dir / file_path
        logger.debug(f"Checking file: {full_path}")
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            logger.debug(f"Found file: {file_path}")
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False
    
    return True

def check_input_file(file_path):
    """Check if input file exists and is valid JSON."""
    logger.debug(f"Checking input file: {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"Input file not found: {file_path}")
            return False
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Check required fields in input data
        required_fields = ['generated_posts']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            logger.error(f"Missing required fields in input JSON: {missing_fields}")
            return False
        
        logger.debug(f"Input file is valid JSON with {len(data['generated_posts'])} posts")
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking input file: {e}")
        return False

def check_pipeline_imports():
    """Check if all required modules can be imported."""
    logger.debug("Checking pipeline imports...")
    try:
        logger.debug("Attempting to import EvaluationPipeline...")
        from research_case.evaluator.pipeline import EvaluationPipeline
        logger.debug("Successfully imported EvaluationPipeline")
        
        logger.debug("Attempting to import RougeEvaluator...")
        from research_case.evaluator.rouge_evaluator import RougeEvaluator
        logger.debug("Successfully imported RougeEvaluator")
        
        logger.debug("Attempting to import SimilarityAnalyzer...")
        from research_case.evaluator.similarity_analyzer import SimilarityAnalyzer
        logger.debug("Successfully imported SimilarityAnalyzer")
        
        logger.debug("Attempting to import LLMJudge...")
        from research_case.evaluator.llm_judge import LLMJudge
        logger.debug("Successfully imported LLMJudge")
        
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Python path:", sys.path)
        return False
    except Exception as e:
        logger.error(f"Error checking imports: {e}")
        logger.error("Python path:", sys.path)
        return False

def main():
    """Run diagnostic checks."""
    logger.info("Starting diagnostic checks...")
    
    # Add project root to Python path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    logger.debug(f"Added project root to Python path: {project_root}")
    
    checks = [
        ("Environment variables", check_environment()),
        ("File structure", check_file_structure()),
        ("Pipeline imports", check_pipeline_imports())
    ]
    
    # Input file check (if path provided)
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        checks.append(("Input file", check_input_file(input_path)))
    
    # Print results
    print("\nDiagnostic Results:")
    print("-" * 50)
    all_passed = True
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\nAll checks passed! If the pipeline still isn't running, check debug.log for more details.")
    else:
        print("\nSome checks failed. Please fix the issues above and try again.")

if __name__ == "__main__":
    main()