#!/usr/bin/env python
"""
Quick script to run the benchmark from the root directory with hard-coded parameters.
"""
import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the benchmark function
from src.benchmark_main import compare_approaches

def main():
    # Hard-coded benchmark parameters
    MAX_DISEASES = 10          # Get top 10 diagnoses for comparison
    MAX_QUESTIONS = 10          # Number of questions each approach can ask
    MAX_WORKERS = 4            # Number of concurrent threads (increased to 4)
    SAVE_RESULTS = True        # Save detailed results to file
    TURNS_BEFORE_NARROWING = 3 # Start narrowing after this many questions
    PARALLEL_CASES = True      # Enable parallel case processing
    CONFIDENCE_THRESHOLD = 0.75 # Confidence threshold for diagnosis
    
    # Run the benchmark with fixed parameters
    compare_approaches(
        max_diseases=MAX_DISEASES,
        max_questions=MAX_QUESTIONS,
        max_workers=MAX_WORKERS,
        save_results=SAVE_RESULTS,
        turns_before_narrowing=TURNS_BEFORE_NARROWING,
        parallel_cases=PARALLEL_CASES,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

if __name__ == "__main__":
    main() 