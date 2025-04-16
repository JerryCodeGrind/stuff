import json
import os
import glob
from typing import Dict, List, Any

def analyze_results(summary_file: str) -> Dict[str, Any]:
    """
    Analyze the incremental summary data and compare IGN vs GPT Doctor.
    
    Args:
        summary_file: Path to the incremental summary JSON file
        
    Returns:
        Dictionary with analysis results
    """
    # Load the summary data
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # Extract parameters
    parameters = data.get('parameters', {})
    
    # Extract cases
    cases = data.get('cases', [])
    
    # Initialize counters
    total_cases = len(cases)
    ign_correct = 0
    gpt_correct = 0
    ign_questions_total = 0
    gpt_questions_total = 0
    ign_top_1 = 0  # Count of ground truth diagnoses ranked #1
    gpt_top_1 = 0
    ign_top_3 = 0  # Count of ground truth diagnoses ranked in top 3
    gpt_top_3 = 0
    ign_top_10 = 0  # Count of ground truth diagnoses ranked in top 10
    gpt_top_10 = 0
    ign_narrowed_out = 0
    gpt_narrowed_out = 0
    
    # Iterate through cases
    for case in cases:
        # Extract diagnosis information
        diagnosis = case.get('diagnosis', '')
        
        # IGN results
        ign_data = case.get('ign', {})
        ign_correct += 1 if ign_data.get('correct_diagnosis', False) else 0
        ign_questions_total += ign_data.get('questions_asked', 0)
        ign_narrowed_out += 1 if ign_data.get('narrowed_out', False) else 0
        
        # Check if ground truth is in top rankings
        ign_rank = ign_data.get('final_rank')
        if ign_rank is not None:
            if ign_rank == 1:
                ign_top_1 += 1
            if ign_rank <= 3:
                ign_top_3 += 1
            if ign_rank <= 10:
                ign_top_10 += 1
        
        # GPT Doctor results
        gpt_data = case.get('gpt', {})
        gpt_correct += 1 if gpt_data.get('correct_diagnosis', False) else 0
        gpt_questions_total += gpt_data.get('questions_asked', 0)
        gpt_narrowed_out += 1 if gpt_data.get('narrowed_out', False) else 0
        
        # Check if ground truth is in top rankings
        gpt_rank = gpt_data.get('final_rank')
        if gpt_rank is not None:
            if gpt_rank == 1:
                gpt_top_1 += 1
            if gpt_rank <= 3:
                gpt_top_3 += 1
            if gpt_rank <= 10:
                gpt_top_10 += 1
    
    # Calculate averages
    ign_avg_questions = ign_questions_total / total_cases if total_cases > 0 else 0
    gpt_avg_questions = gpt_questions_total / total_cases if total_cases > 0 else 0
    
    # Compile analysis results
    results = {
        "summary_file": summary_file,
        "timestamp": data.get('timestamp', ''),
        "parameters": parameters,
        "total_cases": total_cases,
        "ign": {
            "correct_diagnoses": ign_correct,
            "correct_percentage": (ign_correct / total_cases * 100) if total_cases > 0 else 0,
            "avg_questions": ign_avg_questions,
            "top_1_diagnoses": ign_top_1,
            "top_1_percentage": (ign_top_1 / total_cases * 100) if total_cases > 0 else 0,
            "top_3_diagnoses": ign_top_3,
            "top_3_percentage": (ign_top_3 / total_cases * 100) if total_cases > 0 else 0,
            "top_10_diagnoses": ign_top_10,
            "top_10_percentage": (ign_top_10 / total_cases * 100) if total_cases > 0 else 0,
            "narrowed_out_count": ign_narrowed_out,
            "narrowed_out_percentage": (ign_narrowed_out / total_cases * 100) if total_cases > 0 else 0,
        },
        "gpt": {
            "correct_diagnoses": gpt_correct,
            "correct_percentage": (gpt_correct / total_cases * 100) if total_cases > 0 else 0,
            "avg_questions": gpt_avg_questions,
            "top_1_diagnoses": gpt_top_1,
            "top_1_percentage": (gpt_top_1 / total_cases * 100) if total_cases > 0 else 0,
            "top_3_diagnoses": gpt_top_3,
            "top_3_percentage": (gpt_top_3 / total_cases * 100) if total_cases > 0 else 0,
            "top_10_diagnoses": gpt_top_10,
            "top_10_percentage": (gpt_top_10 / total_cases * 100) if total_cases > 0 else 0,
            "narrowed_out_count": gpt_narrowed_out,
            "narrowed_out_percentage": (gpt_narrowed_out / total_cases * 100) if total_cases > 0 else 0,
        },
        "comparison": {
            "correct_diagnosis_difference": ign_correct - gpt_correct,
            "avg_questions_difference": ign_avg_questions - gpt_avg_questions,
            "better_correct_diagnosis": "IGN" if ign_correct > gpt_correct else "GPT" if gpt_correct > ign_correct else "Equal",
            "better_top_1": "IGN" if ign_top_1 > gpt_top_1 else "GPT" if gpt_top_1 > ign_top_1 else "Equal",
            "better_top_3": "IGN" if ign_top_3 > gpt_top_3 else "GPT" if gpt_top_3 > ign_top_3 else "Equal",
            "better_top_10": "IGN" if ign_top_10 > gpt_top_10 else "GPT" if gpt_top_10 > ign_top_10 else "Equal",
            "fewer_questions": "IGN" if ign_avg_questions < gpt_avg_questions else "GPT" if gpt_avg_questions < ign_avg_questions else "Equal",
        }
    }
    
    return results

def print_analysis(analysis: Dict[str, Any]) -> None:
    """Print a formatted summary of the analysis results"""
    print("\n===== INCREMENTAL BENCHMARK ANALYSIS =====")
    print(f"Summary file: {analysis['summary_file']}")
    print(f"Timestamp: {analysis['timestamp']}")
    print(f"Total cases processed: {analysis['total_cases']}")
    
    print("\nParameters:")
    for key, value in analysis['parameters'].items():
        print(f"  {key}: {value}")
    
    print("\nInformation Gain Network (IGN) Results:")
    ign = analysis['ign']
    print(f"  Correct diagnoses: {ign['correct_diagnoses']}/{analysis['total_cases']} ({ign['correct_percentage']:.1f}%)")
    print(f"  Top-1 diagnoses: {ign['top_1_diagnoses']}/{analysis['total_cases']} ({ign['top_1_percentage']:.1f}%)")
    print(f"  Top-3 diagnoses: {ign['top_3_diagnoses']}/{analysis['total_cases']} ({ign['top_3_percentage']:.1f}%)")
    print(f"  Top-10 diagnoses: {ign['top_10_diagnoses']}/{analysis['total_cases']} ({ign['top_10_percentage']:.1f}%)")
    print(f"  Average questions per case: {ign['avg_questions']:.2f}")
    print(f"  Ground truth diagnoses narrowed out: {ign['narrowed_out_count']}/{analysis['total_cases']} ({ign['narrowed_out_percentage']:.1f}%)")
    
    print("\nGPT Doctor Results:")
    gpt = analysis['gpt']
    print(f"  Correct diagnoses: {gpt['correct_diagnoses']}/{analysis['total_cases']} ({gpt['correct_percentage']:.1f}%)")
    print(f"  Top-1 diagnoses: {gpt['top_1_diagnoses']}/{analysis['total_cases']} ({gpt['top_1_percentage']:.1f}%)")
    print(f"  Top-3 diagnoses: {gpt['top_3_diagnoses']}/{analysis['total_cases']} ({gpt['top_3_percentage']:.1f}%)")
    print(f"  Top-10 diagnoses: {gpt['top_10_diagnoses']}/{analysis['total_cases']} ({gpt['top_10_percentage']:.1f}%)")
    print(f"  Average questions per case: {gpt['avg_questions']:.2f}")
    print(f"  Ground truth diagnoses narrowed out: {gpt['narrowed_out_count']}/{analysis['total_cases']} ({gpt['narrowed_out_percentage']:.1f}%)")
    
    print("\nComparison:")
    comp = analysis['comparison']
    print(f"  Better at correct diagnosis: {comp['better_correct_diagnosis']}")
    print(f"  Better at ranking ground truth #1: {comp['better_top_1']}")
    print(f"  Better at ranking ground truth in top 3: {comp['better_top_3']}")
    print(f"  Better at ranking ground truth in top 10: {comp['better_top_10']}")
    print(f"  More efficient (fewer questions): {comp['fewer_questions']}")
    
    # Overall winner determination
    points_ign = 0
    points_gpt = 0
    
    # Correct diagnosis points
    if comp['better_correct_diagnosis'] == "IGN":
        points_ign += 3
    elif comp['better_correct_diagnosis'] == "GPT":
        points_gpt += 3
    else:
        points_ign += 1.5
        points_gpt += 1.5
    
    # Top-1 ranking points
    if comp['better_top_1'] == "IGN":
        points_ign += 2
    elif comp['better_top_1'] == "GPT":
        points_gpt += 2
    else:
        points_ign += 1
        points_gpt += 1
    
    # Top-3 ranking points
    if comp['better_top_3'] == "IGN":
        points_ign += 1
    elif comp['better_top_3'] == "GPT":
        points_gpt += 1
    else:
        points_ign += 0.5
        points_gpt += 0.5
    
    # Top-10 ranking points
    if comp['better_top_10'] == "IGN":
        points_ign += 0.5
    elif comp['better_top_10'] == "GPT":
        points_gpt += 0.5
    else:
        points_ign += 0.25
        points_gpt += 0.25
    
    # Efficiency points
    if comp['fewer_questions'] == "IGN":
        points_ign += 1
    elif comp['fewer_questions'] == "GPT":
        points_gpt += 1
    else:
        points_ign += 0.5
        points_gpt += 0.5
    
    # Determine overall winner
    if points_ign > points_gpt:
        overall_winner = "Information Gain Network (IGN)"
    elif points_gpt > points_ign:
        overall_winner = "GPT Doctor"
    else:
        overall_winner = "Tie - Both methods perform equally"
    
    print(f"\nOverall assessment: {overall_winner} is better based on the current incremental results")
    print(f"  IGN points: {points_ign}, GPT points: {points_gpt}")

def find_most_recent_summary() -> str:
    """Find the most recent incremental summary file"""
    summary_files = glob.glob("benchmark_results/incremental_summary_*.json")
    if not summary_files:
        raise FileNotFoundError("No incremental summary files found in benchmark_results directory")
    
    # Sort by modification time (most recent first)
    summary_files.sort(key=os.path.getmtime, reverse=True)
    return summary_files[0]

def main():
    try:
        # Find the most recent summary file
        summary_file = find_most_recent_summary()
        
        # Analyze the results
        analysis = analyze_results(summary_file)
        
        # Print the analysis
        print_analysis(analysis)
        
        # Save the analysis to a file
        output_file = f"benchmark_results/analysis_{analysis['timestamp']}.txt"
        with open(output_file, 'w') as f:
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            print_analysis(analysis)
            sys.stdout = original_stdout
        
        print(f"\nAnalysis saved to: {output_file}")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    main() 