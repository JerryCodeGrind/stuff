import time
import concurrent.futures
from typing import Dict, List, Tuple, Any

from src.agents.diagnoser_agent import DiagnoserAgent
from src.agents.customer_agent import CustomerAgent
from src.agents.probability_agent import ProbabilityAgent
from src.utils.entropy import calculate_entropy, has_confident_diagnosis
from src.utils.information_gain import evaluate_question_info_gain
from src.models.case import Case, DiagnosisResult

def run_information_gain_network(
    case: Dict[str, str],
    max_diseases: int = 5,
    confidence_threshold: float = 0.75,
    max_questions: int = 5,
    turns_before_narrowing: int = 2,
    questions_per_disease: int = 2,
    number_of_scenarios: int = 5,
    max_workers: int = 2  # Reduced from 5 to 2 for better stability
) -> Dict[str, any]:
    """
    Run the Information Gain Network diagnostic algorithm.
    
    Args:
        case: A patient case dictionary with 'doctor_vignette', 'patient_profile', and 'diagnosis'
        max_diseases: Number of top diseases to consider initially
        confidence_threshold: Threshold for confident diagnosis (if using threshold-based stopping)
        max_questions: Maximum number of questions to ask
        turns_before_narrowing: Number of turns before first narrowing of diseases
        questions_per_disease: Number of questions to generate per disease
        number_of_scenarios: Number of response scenarios to generate for each question
        max_workers: Maximum number of concurrent worker threads
        
    Returns:
        Dictionary containing diagnostic results and statistics
    """
    start_time = time.time()
    
    print(f"\n\n=== NEW PATIENT ===")
    print(f"Doctor sees: {case['doctor_vignette']}")
    print(f"Ground truth diagnosis: {case['diagnosis']}")
    
    # Initialize agents
    diagnoser = DiagnoserAgent()
    probability_agent = ProbabilityAgent(questions_per_disease=questions_per_disease)
    customer = CustomerAgent(case['patient_profile'])
    
    # Get ground truth diagnosis in lowercase for comparison
    ground_truth = case["diagnosis"].lower()
    
    results = {
        "questions_asked": 0,
        "diagnoses": [],
        "confident_diagnosis": False,
        "correct_diagnosis": False,
        "final_diagnosis": None,
        "final_probability": 0.0,
        "ground_truth": case["diagnosis"],
        "narrowing_events": [],
        "ground_truth_rank": None,
        "ground_truth_rank_history": [],
        "ground_truth_narrowed_out": False,
        "ground_truth_last_rank": None
    }
    
    # Get initial probabilities
    print("Calculating initial diagnosis...")
    current_probs, initial_reasoning = diagnoser.update_probabilities(
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
    
    # Continue asking questions until only one disease remains or max questions reached
    questions_asked = 0
    asked_questions = set()  # Track which questions have been asked
    
    # Track narrowing process
    performed_first_narrowing = False
    narrow_count = 0
    focused_diseases = list(current_probs.keys())  # Start with all diseases
    min_diseases_to_keep = 1  # We'll narrow down to just one disease
    
    while len(current_probs) > min_diseases_to_keep and questions_asked < max_questions:
        # Check if we should perform disease narrowing
        should_narrow = False
        
        # Initial narrowing after a certain number of turns
        if not performed_first_narrowing and questions_asked >= turns_before_narrowing:
            should_narrow = True
            performed_first_narrowing = True
        # Subsequent narrowing every 2 questions
        elif performed_first_narrowing and (questions_asked - turns_before_narrowing) % 2 == 0 and len(current_probs) > min_diseases_to_keep:
            should_narrow = True
        
        # Perform narrowing if conditions are met
        if should_narrow:
            narrow_count += 1
            print(f"\n=== PERFORMING DISEASE NARROWING ({narrow_count}) ===")
            
            # Sort diseases by current probability
            sorted_diseases = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate how many diseases to keep - reduce by half each time, minimum 1
            num_to_keep = max(min_diseases_to_keep, len(current_probs) // 2)
            
            # Select top diseases
            focused_diseases = [disease for disease, _ in sorted_diseases[:num_to_keep]]
            
            print(f"Narrowing down to top {num_to_keep} diseases: {', '.join(focused_diseases)}")
            
            # Check if ground truth has been narrowed out
            ground_truth_included = any(ground_truth in disease.lower() for disease in focused_diseases)
            if not ground_truth_included and not results["ground_truth_narrowed_out"]:
                print(f"⚠️ Ground truth diagnosis '{case['diagnosis']}' has been narrowed out!")
                results["ground_truth_narrowed_out"] = True
                # Save the rank before narrowing as the last rank
                if ground_truth_rank is not None:
                    results["ground_truth_last_rank"] = ground_truth_rank
                # Since the ground truth was narrowed out, add None to the rank history
                results["ground_truth_rank_history"].append(None)
            
            # Record the narrowing event
            results["narrowing_events"].append({
                "turn": questions_asked,
                "diseases_before": len(current_probs),
                "diseases_after": num_to_keep,
                "focused_diseases": focused_diseases,
                "ground_truth_included": ground_truth_included
            })
            
            # If narrowed to a single disease, consider it a confident diagnosis
            if num_to_keep == min_diseases_to_keep:
                print(f"Narrowed down to single diagnosis: {focused_diseases[0]}")
                results["confident_diagnosis"] = True
                
                # If only one disease remains, we can exit the loop
                if len(focused_diseases) == 1:
                    print("Single diagnosis reached. Stopping questions.")
                    break
            
            # Generate focused questions for the narrowed set of diseases
            print(f"Generating new focused questions specific to: {', '.join(focused_diseases)}")
            
            # Simple prompt to generate questions focused on the diseases we're considering
            focused_prompt = f"""Generate {questions_per_disease * len(focused_diseases) * 3} HIGHLY SPECIFIC diagnostic questions that would help differentiate ONLY between these specific diseases: {', '.join(focused_diseases)}.
            
            The questions must be STRICTLY focused on distinguishing between these diseases and should NOT ask about unrelated conditions.
            
            Each question should:
            1. Target specific symptoms, signs, or risk factors that help distinguish between these specific diseases
            2. Have high discriminative value between these diseases
            3. Be directly relevant to the differential diagnosis of these conditions
            4. Focus on the characteristics that are MOST different between these specific conditions
            5. Include questions about timing, severity, triggers, and associated symptoms
            
            Return ONLY the questions, one per line, with no additional text or numbering.
            """
            
            response = probability_agent._generate_category_questions(focused_prompt)
            
            # Replace the question set with ONLY the focused questions rather than extending it
            # This ensures we only ask questions relevant to the narrowed diseases
            probability_agent.question_set = response
            
            # Reset the asked questions for the next phase
            asked_questions = set()
            
            print(f"Generated {len(response)} new focused questions specific to: {', '.join(focused_diseases)}")
            
            # Update current_probs to contain only the focused diseases
            # This ensures we're only tracking probabilities for diseases we care about
            focused_probs = {d: current_probs.get(d, 0.0) for d in focused_diseases}
            # Normalize to ensure they sum to 1.0
            total = sum(focused_probs.values())
            if total > 0:
                focused_probs = {d: p/total for d, p in focused_probs.items()}
            current_probs = focused_probs
        
        # Calculate current entropy
        current_entropy = calculate_entropy(tuple(sorted(current_probs.items())))
        
        # Get available questions
        available_questions = [q for q in probability_agent.question_set if q not in asked_questions]
        if not available_questions:
            print("No more questions available.")
            break
        
        # Pick the next best question - evaluate ALL available questions in parallel
        print("\nFinding the best question to ask...")
        
        # Don't limit the number of questions to evaluate - evaluate all of them
        questions_to_evaluate = available_questions
        
        # Use concurrent processing to evaluate questions in parallel, but with a limited number of workers
        question_info_gains = []
        
        # For small numbers of questions, just evaluate sequentially to avoid thread overhead
        if len(questions_to_evaluate) <= 2:
            for question in questions_to_evaluate:
                result = evaluate_question_info_gain(
                    question, diagnoser, current_probs, current_entropy,
                    probability_agent, min(3, number_of_scenarios), max_diseases,  # Reduced number of scenarios
                    performed_first_narrowing, focused_diseases
                )
                question_info_gains.append(result)
        else:
            # For more questions, use parallel processing but with limited workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks
                futures = []
                for question in questions_to_evaluate:
                    futures.append(
                        executor.submit(
                            evaluate_question_info_gain,
                            question, diagnoser, current_probs, current_entropy,
                            probability_agent, min(3, number_of_scenarios), max_diseases,  # Reduced number of scenarios
                            performed_first_narrowing, focused_diseases
                        )
                    )
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        question_info_gains.append(result)
                        # Remove early stopping to evaluate all questions
                    except Exception as e:
                        print(f"Error evaluating question: {e}")
                        continue
        
        # Find the best question
        if question_info_gains:
            best_question, best_info_gain = max(question_info_gains, key=lambda x: x[1])
        else:
            print("No question provides positive information gain. Stopping.")
            break
        
        # Ask the best question
        questions_asked += 1
        asked_questions.add(best_question)
        print(f"\n--- Question {questions_asked} (IG: {best_info_gain:.4f}) ---")
        print(f"Doctor: {best_question}")
        
        # Get patient response
        patient_response = customer.respond_to_question(best_question)
        print(f"Patient: {patient_response}")
        
        # Update diagnosis with the new information
        current_probs, _ = diagnoser.update_probabilities(
            f"Question: {best_question}, Answer: {patient_response}", 
            get_reasoning=True,
            num_diseases=max_diseases if not performed_first_narrowing else len(focused_diseases)
        )
        
        # Filter probabilities to only include focused diseases if narrowing has occurred
        if performed_first_narrowing:
            focused_probs = {d: current_probs.get(d, 0.0) for d in focused_diseases}
            # Normalize to ensure they sum to 1.0
            total = sum(focused_probs.values())
            if total > 0:
                focused_probs = {d: p/total for d, p in focused_probs.items()}
            current_probs = focused_probs
        
        # Update ground truth rank
        if not results["ground_truth_narrowed_out"]:
            ground_truth_rank = None
            sorted_diagnoses = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
            for idx, (diagnosis, _) in enumerate(sorted_diagnoses):
                if ground_truth in diagnosis.lower():
                    ground_truth_rank = idx + 1
                    break
            results["ground_truth_rank"] = ground_truth_rank
            results["ground_truth_rank_history"].append(ground_truth_rank)
            print(f"Current ground truth rank: {ground_truth_rank if ground_truth_rank else 'not in top diseases'}")
        
        # Show updated probabilities
        print(f"\nUpdated disease probabilities:")
        for disease, prob in sorted(current_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {disease}: {prob:.3f}")
        
        # Track the diagnoses at this turn
        results["diagnoses"].append({
            "turn": questions_asked,
            "question": best_question,
            "patient_response": patient_response,
            "probabilities": current_probs,
            "narrowed_diseases": len(current_probs)
        })
    
    # Show final diagnosis
    print("\n=== FINAL DIAGNOSIS ===")
    top_disease, top_prob = max(current_probs.items(), key=lambda x: x[1])
    
    if len(current_probs) == 1 or top_prob >= confidence_threshold:  # Single disease or high probability
        print(f"Confident diagnosis: {top_disease} ({top_prob:.3f})")
        results["confident_diagnosis"] = True
    else:
        print(f"Diagnosis uncertain. Most likely: {top_disease} ({top_prob:.3f})")
    
    # Check if diagnosis matches ground truth
    correct = case["diagnosis"].lower() in top_disease.lower()
    print(f"Correct diagnosis: {correct} (Ground truth: {case['diagnosis']})")
    
    # Add final statistics
    results["questions_asked"] = questions_asked
    results["final_diagnosis"] = top_disease
    results["final_probability"] = top_prob
    results["correct_diagnosis"] = correct
    results["final_disease_count"] = len(current_probs)
    results["narrowing_steps"] = narrow_count
    
    # Ensure final ground truth rank is properly recorded
    if not results["ground_truth_narrowed_out"]:
        # Ground truth not narrowed out, calculate final rank
        ground_truth_rank = None
        sorted_diagnoses = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        for idx, (diagnosis, _) in enumerate(sorted_diagnoses):
            if ground_truth in diagnosis.lower():
                ground_truth_rank = idx + 1
                break
        results["ground_truth_rank"] = ground_truth_rank
        
        # Ensure rank_history is complete (in case we missed any updates)
        if len(results["ground_truth_rank_history"]) <= questions_asked:
            # Add current rank to history to match questions asked
            results["ground_truth_rank_history"].append(ground_truth_rank)
    else:
        # Ground truth was narrowed out, use the last known rank
        print(f"Note: Ground truth was narrowed out. Last rank before narrowing: {results['ground_truth_last_rank']}")
    
    print(f"Questions asked: {questions_asked}/{max_questions}")
    print(f"Final disease count: {len(current_probs)}")
    print(f"Narrowing steps: {narrow_count}")
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    return results