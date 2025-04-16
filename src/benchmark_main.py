import time
from typing import Dict, List, Any
import json
import os
import sys
import concurrent.futures

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.case import cases
from src.runners.diagnostic_engine import run_information_gain_network
from src.runners.benchmark import run_benchmark_suite, run_gpt_doctor_benchmark

def compare_approaches(
    max_diseases: int = 10,
    max_questions: int = 5,
    max_workers: int = 2,
    save_results: bool = True,
    turns_before_narrowing: int = 2,
    parallel_cases: bool = True,
    confidence_threshold: float = 0.75
):
    """
    Run both approaches (IGN and GPT Doctor) and compare their performance
    
    Args:
        max_diseases: Maximum number of diseases to track (set to 10 for top 10 diagnoses)
        max_questions: Maximum number of questions to ask
        max_workers: Maximum number of concurrent worker threads
        save_results: Whether to save results to a file
        turns_before_narrowing: Number of turns before first narrowing of diseases
        parallel_cases: Whether to process cases in parallel
        confidence_threshold: Probability threshold for confident diagnosis
        
    Returns:
        Comparison dictionary
    """
    print("\n=== BENCHMARKING COMPARISON ===")
    print(f"Max diseases: {max_diseases}")
    print(f"Max questions: {max_questions}")
    print(f"Max worker threads: {max_workers}")
    print(f"Turns before narrowing: {turns_before_narrowing}")
    print(f"Parallel case processing: {parallel_cases}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("-" * 50)
    
    # Set common parameters
    ign_results = []
    gpt_results = []
    overall_start = time.time()
    
    # Define a function to process a single case
    def process_case(case_idx, case):
        case_start_time = time.time()
        print(f"\n--- CASE {case_idx+1}/{len(cases)} ---")
        
        # Run Information Gain Network approach
        print("\nRunning Information Gain Network approach...")
        ign_result = run_information_gain_network(
            case,
            max_diseases=max_diseases,
            max_questions=max_questions,
            max_workers=max_workers,
            turns_before_narrowing=turns_before_narrowing,
            confidence_threshold=confidence_threshold
        )
        
        # Run GPT Doctor approach
        print("\nRunning GPT Doctor approach...")
        gpt_result = run_gpt_doctor_benchmark(
            case,
            max_diseases=max_diseases,
            max_questions=max_questions,
            confidence_threshold=confidence_threshold
        )
        
        case_time = time.time() - case_start_time
        print(f"Case {case_idx+1} completed in {case_time:.2f} seconds")
        
        # Save incremental results after each case
        save_incremental_results(case_idx, case, ign_result, gpt_result)
        
        return ign_result, gpt_result
    
    # Helper function to save incremental results
    def save_incremental_results(case_idx, case, ign_result, gpt_result):
        # Create results directory if it doesn't exist
        if not os.path.exists("benchmark_results"):
            os.makedirs("benchmark_results")
            
        # Generate timestamp if not already set
        if not hasattr(save_incremental_results, "timestamp"):
            save_incremental_results.timestamp = time.strftime("%Y%m%d-%H%M%S")
            
        # Prepare case data
        ground_truth = case["diagnosis"].lower()
            
        # Calculate ground truth rank for IGN
        ign_rank = None
        if "ground_truth_rank" in ign_result and ign_result["ground_truth_rank"] is not None:
            ign_rank = ign_result["ground_truth_rank"]
        elif "ground_truth_narrowed_out" in ign_result and ign_result["ground_truth_narrowed_out"] and "ground_truth_last_rank" in ign_result:
            ign_rank = ign_result["ground_truth_last_rank"] if ign_result["ground_truth_last_rank"] is not None else 999
        else:
            # Calculate from final probabilities
            ign_final_diagnoses = ign_result["diagnoses"][-1]["probabilities"]
            ign_sorted_diagnoses = sorted(ign_final_diagnoses.items(), key=lambda x: x[1], reverse=True)
            ign_rank = 999  # Default if not found
            
            for idx, (diagnosis, _) in enumerate(ign_sorted_diagnoses):
                if ground_truth in diagnosis.lower():
                    ign_rank = idx + 1
                    break
                    
        # Calculate ground truth rank for GPT
        gpt_rank = None
        if "ground_truth_rank" in gpt_result and gpt_result["ground_truth_rank"] is not None:
            gpt_rank = gpt_result["ground_truth_rank"]
        elif "ground_truth_narrowed_out" in gpt_result and gpt_result["ground_truth_narrowed_out"] and "ground_truth_last_rank" in gpt_result:
            gpt_rank = gpt_result["ground_truth_last_rank"] if gpt_result["ground_truth_last_rank"] is not None else 999
        else:
            # Calculate from final probabilities
            gpt_final_diagnoses = gpt_result["diagnoses"][-1]["probabilities"]
            gpt_sorted_diagnoses = sorted(gpt_final_diagnoses.items(), key=lambda x: x[1], reverse=True)
            gpt_rank = 999  # Default if not found
            
            for idx, (diagnosis, _) in enumerate(gpt_sorted_diagnoses):
                if ground_truth in diagnosis.lower():
                    gpt_rank = idx + 1
                    break
            
        # Create case data
        case_data = {
            "case_id": case_idx + 1,
            "diagnosis": case["diagnosis"],
            "ign": {
                "questions_asked": ign_result["questions_asked"],
                "final_rank": ign_rank if ign_rank < 999 else None,
                "rank_history": ign_result.get("ground_truth_rank_history", []),
                "narrowed_out": ign_result.get("ground_truth_narrowed_out", False),
                "last_rank_before_narrowing": ign_result.get("ground_truth_last_rank"),
                "correct_diagnosis": ign_result["correct_diagnosis"]
            },
            "gpt": {
                "questions_asked": gpt_result["questions_asked"],
                "final_rank": gpt_rank if gpt_rank < 999 else None,
                "rank_history": gpt_result.get("ground_truth_rank_history", []),
                "narrowed_out": gpt_result.get("ground_truth_narrowed_out", False),
                "last_rank_before_narrowing": gpt_result.get("ground_truth_last_rank"),
                "correct_diagnosis": gpt_result["correct_diagnosis"]
            }
        }
        
        # Save to incremental file
        incremental_file = f"benchmark_results/incremental_case_{case_idx+1}_{save_incremental_results.timestamp}.json"
        with open(incremental_file, 'w') as f:
            json.dump(case_data, f, indent=2)
            
        print(f"Incremental results for case {case_idx+1} saved to: {incremental_file}")
        
        # Also save an updated summary file
        summary_file = f"benchmark_results/incremental_summary_{save_incremental_results.timestamp}.json"
        
        # Load existing summary if it exists, otherwise create new
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
                
            # Update the cases list
            for i, existing_case in enumerate(summary_data["cases"]):
                if existing_case["case_id"] == case_idx + 1:
                    summary_data["cases"][i] = case_data
                    break
            else:
                summary_data["cases"].append(case_data)
        else:
            # Create new summary
            summary_data = {
                "timestamp": save_incremental_results.timestamp,
                "parameters": {
                    "max_diseases": max_diseases,
                    "max_questions": max_questions,
                    "turns_before_narrowing": turns_before_narrowing,
                    "confidence_threshold": confidence_threshold
                },
                "cases": [case_data]
            }
            
        # Save updated summary
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        print(f"Updated summary saved to: {summary_file}")
    
    # Run cases either in parallel or sequentially
    if parallel_cases and len(cases) > 1:
        # Process cases in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(cases))) as executor:
            # Submit tasks
            futures = []
            for i, case in enumerate(cases):
                futures.append(executor.submit(process_case, i, case))
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                ign_result, gpt_result = future.result()
                ign_results.append(ign_result)
                gpt_results.append(gpt_result)
    else:
        # Process cases sequentially
        for i, case in enumerate(cases):
            ign_result, gpt_result = process_case(i, case)
            ign_results.append(ign_result)
            gpt_results.append(gpt_result)
    
    # Calculate overall statistics for IGN
    ign_correct = sum(1 for r in ign_results if r["correct_diagnosis"])
    ign_confident = sum(1 for r in ign_results if r["confident_diagnosis"])
    ign_questions = sum(r["questions_asked"] for r in ign_results)
    
    # Calculate overall statistics for GPT Doctor
    gpt_correct = sum(1 for r in gpt_results if r["correct_diagnosis"])
    gpt_confident = sum(1 for r in gpt_results if r["confident_diagnosis"])
    gpt_questions = sum(r["questions_asked"] for r in gpt_results)
    
    # Calculate ground truth ranking statistics
    ign_rankings = []
    gpt_rankings = []
    
    for i, case in enumerate(cases):
        ground_truth = case["diagnosis"].lower()
        
        # Get the ground truth rank directly from the results if available
        # This handles cases where tracking was done throughout the process
        if "ground_truth_rank" in ign_results[i] and ign_results[i]["ground_truth_rank"] is not None:
            ign_rank = ign_results[i]["ground_truth_rank"]
        elif "ground_truth_narrowed_out" in ign_results[i] and ign_results[i]["ground_truth_narrowed_out"] and "ground_truth_last_rank" in ign_results[i]:
            # If narrowed out, use the last rank before narrowing
            ign_rank = ign_results[i]["ground_truth_last_rank"] if ign_results[i]["ground_truth_last_rank"] is not None else 999
        else:
            # Fall back to calculating from final probabilities
            ign_final_diagnoses = ign_results[i]["diagnoses"][-1]["probabilities"]
            ign_sorted_diagnoses = sorted(ign_final_diagnoses.items(), key=lambda x: x[1], reverse=True)
            ign_rank = 999  # Default if not found
            
            for idx, (diagnosis, _) in enumerate(ign_sorted_diagnoses):
                if ground_truth in diagnosis.lower():
                    ign_rank = idx + 1
                    break
        
        ign_rankings.append(ign_rank)
        
        # Same logic for GPT Doctor
        if "ground_truth_rank" in gpt_results[i] and gpt_results[i]["ground_truth_rank"] is not None:
            gpt_rank = gpt_results[i]["ground_truth_rank"]
        elif "ground_truth_narrowed_out" in gpt_results[i] and gpt_results[i]["ground_truth_narrowed_out"] and "ground_truth_last_rank" in gpt_results[i]:
            # If narrowed out, use the last rank before narrowing
            gpt_rank = gpt_results[i]["ground_truth_last_rank"] if gpt_results[i]["ground_truth_last_rank"] is not None else 999
        else:
            # Fall back to calculating from final probabilities
            gpt_final_diagnoses = gpt_results[i]["diagnoses"][-1]["probabilities"]
            gpt_sorted_diagnoses = sorted(gpt_final_diagnoses.items(), key=lambda x: x[1], reverse=True)
            gpt_rank = 999  # Default if not found
            
            for idx, (diagnosis, _) in enumerate(gpt_sorted_diagnoses):
                if ground_truth in diagnosis.lower():
                    gpt_rank = idx + 1
                    break
                
        gpt_rankings.append(gpt_rank)
    
    # Compile comparison results
    comparison = {
        "total_cases": len(cases),
        "total_time": time.time() - overall_start,
        "information_gain_network": {
            "correct_diagnoses": ign_correct,
            "correct_percentage": ign_correct/len(cases)*100 if cases else 0,
            "confident_diagnoses": ign_confident,
            "confident_percentage": ign_confident/len(cases)*100 if cases else 0,
            "avg_questions": ign_questions/len(cases) if cases else 0,
            "ground_truth_rankings": ign_rankings,
            "detailed_results": ign_results
        },
        "gpt_doctor": {
            "correct_diagnoses": gpt_correct,
            "correct_percentage": gpt_correct/len(cases)*100 if cases else 0,
            "confident_diagnoses": gpt_confident,
            "confident_percentage": gpt_confident/len(cases)*100 if cases else 0,
            "avg_questions": gpt_questions/len(cases) if cases else 0,
            "ground_truth_rankings": gpt_rankings,
            "detailed_results": gpt_results
        }
    }
    
    # Print comparison
    print("\n=== COMPARISON RESULTS ===")
    print(f"Cases processed: {comparison['total_cases']}")
    print(f"Total execution time: {comparison['total_time']:.2f} seconds")
    
    print("\nInformation Gain Network:")
    print(f"  Correct diagnoses: {comparison['information_gain_network']['correct_diagnoses']}/{comparison['total_cases']} ({comparison['information_gain_network']['correct_percentage']:.1f}%)")
    print(f"  Confident diagnoses: {comparison['information_gain_network']['confident_diagnoses']}/{comparison['total_cases']} ({comparison['information_gain_network']['confident_percentage']:.1f}%)")
    print(f"  Average questions per case: {comparison['information_gain_network']['avg_questions']:.1f}")
    
    print("\nGPT Doctor:")
    print(f"  Correct diagnoses: {comparison['gpt_doctor']['correct_diagnoses']}/{comparison['total_cases']} ({comparison['gpt_doctor']['correct_percentage']:.1f}%)")
    print(f"  Confident diagnoses: {comparison['gpt_doctor']['confident_diagnoses']}/{comparison['total_cases']} ({comparison['gpt_doctor']['confident_percentage']:.1f}%)")
    print(f"  Average questions per case: {comparison['gpt_doctor']['avg_questions']:.1f}")
    
    # Print detailed rankings and top diagnoses for each case
    for i, case in enumerate(cases):
        print(f"\n--- DETAILED RESULTS FOR CASE {i+1} ({case['diagnosis']}) ---")
        
        # Get final diagnoses
        ign_final_diagnoses = ign_results[i]["diagnoses"][-1]["probabilities"]
        ign_sorted_diagnoses = sorted(ign_final_diagnoses.items(), key=lambda x: x[1], reverse=True)
        
        gpt_final_diagnoses = gpt_results[i]["diagnoses"][-1]["probabilities"]
        gpt_sorted_diagnoses = sorted(gpt_final_diagnoses.items(), key=lambda x: x[1], reverse=True)
        
        # Print comparison
        print(f"Ground truth diagnosis: {case['diagnosis']}")
        print(f"IGN questions asked: {ign_results[i]['questions_asked']} (Ground truth rank: {ign_rankings[i] if ign_rankings[i] < 999 else 'not found'})")
        print(f"GPT questions asked: {gpt_results[i]['questions_asked']} (Ground truth rank: {gpt_rankings[i] if gpt_rankings[i] < 999 else 'not found'})")
        
        # Print top 10 diagnoses
        print("\nTop 10 Diagnoses Comparison:")
        print("-" * 80)
        print(f"{'Rank':<5} | {'Information Gain Network':<35} | {'Prob':<6} | {'GPT Doctor':<35} | {'Prob':<6}")
        print("-" * 80)
        
        for j in range(min(10, max(len(ign_sorted_diagnoses), len(gpt_sorted_diagnoses)))):
            ign_diagnosis = ign_sorted_diagnoses[j][0] if j < len(ign_sorted_diagnoses) else ""
            ign_prob = f"{ign_sorted_diagnoses[j][1]:.3f}" if j < len(ign_sorted_diagnoses) else ""
            gpt_diagnosis = gpt_sorted_diagnoses[j][0] if j < len(gpt_sorted_diagnoses) else ""
            gpt_prob = f"{gpt_sorted_diagnoses[j][1]:.3f}" if j < len(gpt_sorted_diagnoses) else ""
            
            # Highlight ground truth diagnosis
            ign_highlight = " *" if ground_truth in ign_diagnosis.lower() else ""
            gpt_highlight = " *" if ground_truth in gpt_diagnosis.lower() else ""
            
            print(f"{j+1:<5} | {ign_diagnosis[:33] + ign_highlight:<35} | {ign_prob:<6} | {gpt_diagnosis[:33] + gpt_highlight:<35} | {gpt_prob:<6}")
    
    if save_results:
        # Create results directory if it doesn't exist
        if not os.path.exists("benchmark_results"):
            os.makedirs("benchmark_results")
        
        # Generate timestamp for filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save detailed comparison to file
        results_file = f"benchmark_results/comparison_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        # Save ground truth rank data to a separate file
        rank_data = {
            "timestamp": timestamp,
            "parameters": {
                "max_diseases": max_diseases,
                "max_questions": max_questions,
                "turns_before_narrowing": turns_before_narrowing,
                "confidence_threshold": confidence_threshold
            },
            "raw_data": {
                "ign_rankings": ign_rankings,
                "gpt_rankings": gpt_rankings,
                "correct_diagnoses": {
                    "ign": [r["correct_diagnosis"] for r in ign_results],
                    "gpt": [r["correct_diagnosis"] for r in gpt_results]
                },
                "narrowed_out": {
                    "ign": [r.get("ground_truth_narrowed_out", False) for r in ign_results],
                    "gpt": [r.get("ground_truth_narrowed_out", False) for r in gpt_results]
                },
                "last_ranks_before_narrowing": {
                    "ign": [r.get("ground_truth_last_rank") for r in ign_results],
                    "gpt": [r.get("ground_truth_last_rank") for r in gpt_results]
                },
                "rank_histories": {
                    "ign": [r.get("ground_truth_rank_history", []) for r in ign_results],
                    "gpt": [r.get("ground_truth_rank_history", []) for r in gpt_results]
                }
            },
            "cases": []
        }
        
        for i, case in enumerate(cases):
            # Get final diagnoses and ranks
            ign_result = ign_results[i]
            gpt_result = gpt_results[i]
            
            # Get the current ranks (already calculated above)
            current_ign_rank = ign_rankings[i]
            current_gpt_rank = gpt_rankings[i]
            
            # Collect detailed rank information
            case_rank_data = {
                "case_id": i + 1,
                "diagnosis": case["diagnosis"],
                "ign": {
                    "questions_asked": ign_result["questions_asked"],
                    "final_rank": current_ign_rank if current_ign_rank < 999 else None,
                    "rank_history": ign_result.get("ground_truth_rank_history", []),
                    "narrowed_out": ign_result.get("ground_truth_narrowed_out", False),
                    "last_rank_before_narrowing": ign_result.get("ground_truth_last_rank")
                },
                "gpt": {
                    "questions_asked": gpt_result["questions_asked"],
                    "final_rank": current_gpt_rank if current_gpt_rank < 999 else None,
                    "rank_history": gpt_result.get("ground_truth_rank_history", []),
                    "narrowed_out": gpt_result.get("ground_truth_narrowed_out", False),
                    "last_rank_before_narrowing": gpt_result.get("ground_truth_last_rank")
                }
            }
            rank_data["cases"].append(case_rank_data)
        
        # Save to a separate file
        rank_file = f"benchmark_results/ground_truth_ranks_{timestamp}.json"
        with open(rank_file, 'w') as f:
            json.dump(rank_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Ground truth rank data saved to: {rank_file}")
    
    return comparison

def main():
    """Main entry point for benchmark comparisons"""
    # Hard-coded benchmark parameters
    MAX_DISEASES = 10          # Get top 10 diagnoses for comparison
    MAX_QUESTIONS = 10          # Number of questions each approach can ask
    MAX_WORKERS = 4            # Number of concurrent threads
    SAVE_RESULTS = True        # Save detailed results to file
    TURNS_BEFORE_NARROWING = 1 # Start narrowing after this many questions
    PARALLEL_CASES = True      # Enable parallel case processing
    CONFIDENCE_THRESHOLD = 0.75 # Confidence threshold for diagnosis
    
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