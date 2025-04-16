import time
from typing import Dict, List, Any
import concurrent.futures

from src.models.case import Case, DiagnosisResult
from src.agents.gpt_doctor_agent import GPTDoctorAgent
from src.agents.customer_agent import CustomerAgent

def run_gpt_doctor_benchmark(
    case: Dict[str, str],
    max_diseases: int = 10,
    max_questions: int = 5,
    confidence_threshold: float = 0.75
) -> Dict[str, any]:
    """
    Run a benchmark test using the GPT doctor agent for diagnosis.
    
    This simulates a traditional doctor-patient interaction where the doctor
    asks questions sequentially based on their medical training, not using
    information gain optimization.
    
    Args:
        case: A patient case dictionary with doctor_vignette, patient_profile, and diagnosis
        max_diseases: Number of top diseases to consider
        max_questions: Maximum number of questions to ask
        confidence_threshold: Probability threshold for confident diagnosis (0.75 by default)
        
    Returns:
        Dictionary containing diagnostic results and statistics
    """
    start_time = time.time()
    
    print(f"\n\n=== NEW PATIENT (GPT DOCTOR BENCHMARK) ===")
    print(f"Doctor sees: {case['doctor_vignette']}")
    print(f"Ground truth diagnosis: {case['diagnosis']}")
    
    # Initialize agents
    doctor = GPTDoctorAgent()
    customer = CustomerAgent(case['patient_profile'])
    
    # Get ground truth diagnosis in lowercase for comparison
    ground_truth = case["diagnosis"].lower()
    
    # Track results
    results = {
        "questions_asked": 0,
        "diagnoses": [],
        "confident_diagnosis": False,
        "correct_diagnosis": False,
        "final_diagnosis": None,
        "final_probability": 0.0,
        "ground_truth": case["diagnosis"],
        "interaction_history": [],
        "ground_truth_rank": None,
        "ground_truth_rank_history": [],
        "ground_truth_narrowed_out": False,
        "ground_truth_last_rank": None
    }
    
    # Get initial probabilities
    print("Calculating initial diagnosis...")
    current_probs, _ = doctor.update_probabilities(
        case['doctor_vignette'], 
        get_reasoning=True,
        num_diseases=max_diseases
    )
    
    print("\n=== INITIAL PROBABILITIES ===")
    for disease, prob in sorted(current_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {disease}: {prob:.3f}")
    
    # Track the initial diagnoses
    results["diagnoses"].append({
        "turn": 0,
        "probabilities": current_probs
    })
    
    # Calculate and track initial ground truth rank
    ground_truth_rank = None
    sorted_diagnoses = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
    for idx, (diagnosis, _) in enumerate(sorted_diagnoses):
        if ground_truth in diagnosis.lower():
            ground_truth_rank = idx + 1
            break
    
    results["ground_truth_rank"] = ground_truth_rank
    results["ground_truth_rank_history"].append(ground_truth_rank)
    print(f"Initial ground truth rank: {ground_truth_rank}")
    
    # Ask questions until max reached
    questions_asked = 0
    
    while questions_asked < max_questions:
        questions_asked += 1
        
        # Generate the next question
        question, _ = doctor.generate_next_question(get_reasoning=True)
        
        print(f"\n--- Question {questions_asked} ---")
        print(f"Doctor: {question}")
        
        # Get patient response
        patient_response = customer.respond_to_question(question)
        print(f"Patient: {patient_response}")
        
        # Track the interaction
        results["interaction_history"].append({
            "turn": questions_asked,
            "question": question,
            "patient_response": patient_response
        })
        
        # Update diagnosis with the new information
        current_probs, _ = doctor.update_probabilities(
            f"Question: {question}, Answer: {patient_response}", 
            get_reasoning=True,
            num_diseases=max_diseases
        )
        
        # Show updated probabilities
        print(f"\nUpdated disease probabilities:")
        for disease, prob in sorted(current_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {disease}: {prob:.3f}")
        
        # Update ground truth rank
        ground_truth_rank = None
        sorted_diagnoses = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        for idx, (diagnosis, _) in enumerate(sorted_diagnoses):
            if ground_truth in diagnosis.lower():
                ground_truth_rank = idx + 1
                break
        
        results["ground_truth_rank"] = ground_truth_rank
        results["ground_truth_rank_history"].append(ground_truth_rank)
        print(f"Current ground truth rank: {ground_truth_rank if ground_truth_rank else 'not in top diseases'}")
        
        # Track the diagnoses at this turn
        results["diagnoses"].append({
            "turn": questions_asked,
            "question": question,
            "patient_response": patient_response,
            "probabilities": current_probs
        })
        
        # Check if we have a confident diagnosis
        top_disease, top_prob = max(current_probs.items(), key=lambda x: x[1])
        if top_prob >= confidence_threshold:
            print(f"\nConfident diagnosis reached: {top_disease} ({top_prob:.3f})")
            results["confident_diagnosis"] = True
            break
    
    # Show final diagnosis
    print("\n=== FINAL DIAGNOSIS ===")
    top_disease, top_prob = max(current_probs.items(), key=lambda x: x[1])
    
    print(f"Most likely: {top_disease} ({top_prob:.3f})")
    
    # Check if diagnosis matches ground truth
    correct = case["diagnosis"].lower() in top_disease.lower()
    print(f"Correct diagnosis: {correct} (Ground truth: {case['diagnosis']})")
    
    results["questions_asked"] = questions_asked
    results["final_diagnosis"] = top_disease
    results["final_probability"] = top_prob
    results["correct_diagnosis"] = correct
    
    print(f"Questions asked: {questions_asked}/{max_questions}")
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print("=" * 70)
    
    return results

def run_benchmark_suite(
    cases: List[Dict[str, str]],
    max_diseases: int = 10,
    max_questions: int = 5,
    confidence_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Run benchmark tests on a set of cases.
    
    Args:
        cases: List of patient cases
        max_diseases: Number of top diseases to consider
        max_questions: Maximum number of questions to ask
        confidence_threshold: Probability threshold for confident diagnosis
        
    Returns:
        Dictionary of benchmark results
    """
    global_start_time = time.time()
    results = []
    
    for i, case in enumerate(cases):
        print(f"Processing benchmark case {i+1}/{len(cases)}...")
        
        # Run the GPT doctor benchmark
        case_result = run_gpt_doctor_benchmark(
            case,
            max_diseases=max_diseases,
            max_questions=max_questions,
            confidence_threshold=confidence_threshold
        )
        
        results.append(case_result)
    
    # Calculate overall statistics
    correct_diagnoses = sum(1 for r in results if r["correct_diagnosis"])
    confident_diagnoses = sum(1 for r in results if r["confident_diagnosis"])
    total_questions = sum(r["questions_asked"] for r in results)
    total_time = time.time() - global_start_time
    
    # Compile the benchmark summary
    benchmark_summary = {
        "total_cases": len(results),
        "correct_diagnoses": correct_diagnoses,
        "correct_percentage": correct_diagnoses/len(results)*100 if results else 0,
        "confident_diagnoses": confident_diagnoses,
        "confident_percentage": confident_diagnoses/len(results)*100 if results else 0,
        "avg_questions": total_questions/len(results) if results else 0,
        "total_time": total_time,
        "avg_time_per_case": total_time/len(results) if results else 0,
        "detailed_results": results
    }
    
    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Cases processed: {benchmark_summary['total_cases']}")
    print(f"Correct diagnoses: {benchmark_summary['correct_diagnoses']}/{benchmark_summary['total_cases']} ({benchmark_summary['correct_percentage']:.1f}%)")
    print(f"Confident diagnoses: {benchmark_summary['confident_diagnoses']}/{benchmark_summary['total_cases']} ({benchmark_summary['confident_percentage']:.1f}%)")
    print(f"Average questions per case: {benchmark_summary['avg_questions']:.1f}")
    print(f"Total execution time: {benchmark_summary['total_time']:.2f} seconds")
    print(f"Average time per case: {benchmark_summary['avg_time_per_case']:.2f} seconds")
    print("=" * 70)
    
    return benchmark_summary 