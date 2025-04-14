from typing import Dict, Tuple, List, Any
import concurrent.futures
from src.utils.entropy import calculate_entropy

def evaluate_question_info_gain(question, diagnoser, current_probs, current_entropy, 
                              probability_agent, number_of_scenarios, max_diseases, 
                              performed_narrowing=False, focused_diseases=None):
    """Evaluate information gain for a single question"""
    # Calculate scenario probabilities
    scenario_probs = probability_agent.calculate_scenario_probabilities(
        diagnoser.patient_info, question, number_of_scenarios
    )
    
    # Calculate expected information gain
    expected_entropy = 0
    for scenario, prob in scenario_probs.items():
        # Create a temporary diagnoser to calculate updated probabilities
        temp_diagnoser = type(diagnoser)()
        temp_diagnoser.patient_info = diagnoser.patient_info
        temp_diagnoser.previous_probabilities = current_probs
        temp_diagnoser.base_diseases = diagnoser.base_diseases
        
        # Update probabilities based on scenario
        new_probs = temp_diagnoser.update_probabilities(
            f"Question: {question}, Answer: {scenario}",
            num_diseases=max_diseases
        )
        
        # If we've performed narrowing, prioritize information gain for focused diseases
        if performed_narrowing and focused_diseases:
            # Only consider the entropy of the focused diseases
            focused_probs = {d: new_probs.get(d, 0.0) for d in focused_diseases}
            # Normalize these probabilities
            total = sum(focused_probs.values())
            if total > 0:
                focused_probs = {d: p/total for d, p in focused_probs.items()}
            scenario_entropy = calculate_entropy(tuple(sorted(focused_probs.items())))
        else:
            # Calculate entropy for this scenario using all diseases
            scenario_entropy = calculate_entropy(tuple(sorted(new_probs.items())))
        
        expected_entropy += prob * scenario_entropy
    
    # Calculate information gain
    info_gain = current_entropy - expected_entropy
    
    return (question, info_gain) 