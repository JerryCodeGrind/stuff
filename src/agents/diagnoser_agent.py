from typing import Dict, Tuple, List
from src.utils.api import call_completion_api

class DiagnoserAgent:
    def __init__(self):
        self.patient_info = ""
        self.previous_probabilities = {}
        self.base_diseases = None  # Will be set after first update
    
    def update_probabilities(self, additional_info: str, get_reasoning: bool = False, num_diseases: int = 5) -> Dict[str, float]:
        """
        Update probabilities based on new information.
        
        Args:
            additional_info: New clinical information
            get_reasoning: Whether to return reasoning along with probabilities (deprecated, kept for backward compatibility)
            num_diseases: Number of top diseases to include in the response
            
        Returns:
            Dictionary of disease probabilities, and empty reasoning if get_reasoning is True
        """
        prompt = f"""You are an expert medical diagnosis assistant with extensive knowledge of internal medicine, symptomatology, and differential diagnosis. Based on the information provided below, determine the SPECIFIC top {num_diseases} most likely diagnoses along with their updated probabilities.

Patient Information: {self.patient_info}  
Prior Probabilities: {self.previous_probabilities}  
New Clinical Information: {additional_info}

Your task is to update the probabilities using Bayesian reasoning, incorporating new evidence without introducing bias toward the prior probabilities—they are provided only to give context about the patient's prior likelihoods.

### Medical Knowledge Guidelines:
- Consider the epidemiology and prevalence of different conditions
- Analyze the temporal sequence of symptom development
- Evaluate risk factors and comorbidities
- Assess the specificity and sensitivity of reported symptoms
- Factor in demographic information appropriately
- Recognize common symptom patterns and clinical presentations
- Consider both common and rare diagnoses that fit the symptom profile

### Key Instructions:
- Consider how each new piece of information impacts the existing probabilities.
- Avoid mechanical normalization: only adjust probabilities when there's a justifiable reason based on the evidence.
- If a symptom is not relevant to a condition, do not adjust that condition's probability.
- Your output should be explainable—each probability should reflect an evidence-based shift, whether increased, decreased, or unchanged.
- Always use the FULL, proper medical name for each diagnosis (e.g., "Heart Failure" not "Failure", "Chronic Obstructive Pulmonary Disease" not "COPD")
- When new information suggests a diagnosis not previously considered, you should include it and adjust probabilities accordingly.
- You MUST return EXACTLY {num_diseases} diagnoses, even if some have very low probabilities.

### CRITICAL OUTPUT FORMAT:
You MUST return EXACTLY {num_diseases} diagnoses with their probabilities, formatted as follows:

Disease name 1|0.XXX
Disease name 2|0.XXX
Disease name 3|0.XXX
Disease name 4|0.XXX
Disease name 5|0.XXX
(etc. until you have provided exactly {num_diseases} diseases)

Where:
- Each disease-probability pair must be on its own line
- Use the pipe character | between disease name and probability
- Probabilities must be between 0 and 1 (e.g., 0.350)
- The sum of all probabilities must equal 1.000 exactly
- Do not include any narrative text, explanations, or additional information

This exact format is required for automated parsing. Do not deviate from it in any way.
"""
        
        # Make API call
        response = call_completion_api(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content
        
        # Simple, direct parsing approach - just look for lines with pipe symbols
        probabilities = {}
        lines = [line.strip() for line in content.split('\n') if line.strip() and '|' in line]
        
        for line in lines:
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    disease = parts[0].strip()
                    try:
                        prob = float(parts[1].strip())
                        if disease and 0 <= prob <= 1:
                            probabilities[disease] = prob
                    except ValueError:
                        # If we can't convert to float, just skip this line
                        continue
        
        # Update base diseases logic
        if self.base_diseases is None:
            self.base_diseases = list(probabilities.keys())
        else:
            # Add any new diseases to the base list
            for disease in probabilities.keys():
                if disease not in self.base_diseases:
                    self.base_diseases.append(disease)
        
        # Normalize to ensure they sum to exactly 1.0
        total = sum(probabilities.values())
        if total > 0:
            # First normalize all probabilities
            probabilities = {k: v/total for k, v in probabilities.items()}
            # Then adjust the last probability to ensure exact sum of 1.0
            if probabilities:
                last_disease = list(probabilities.keys())[-1]
                current_sum = sum(probabilities.values())
                probabilities[last_disease] += (1.0 - current_sum)
        
        # Ensure we have exactly num_diseases
        if len(probabilities) < num_diseases:
            # Add additional diseases with very small probabilities
            existing_diseases = set(probabilities.keys())
            for disease in ["Warts", "Psoriasis", "Eczema (Atopic Dermatitis)", "Contact Dermatitis", "Seborrheic Keratosis", "Actinic Keratosis", "Lichen Planus", "Folliculitis", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]:
                if disease not in existing_diseases and len(probabilities) < num_diseases:
                    probabilities[disease] = 0.0001
            
            # Normalize again after adding new diseases
            total = sum(probabilities.values())
            if total > 0:
                probabilities = {k: v/total for k, v in probabilities.items()}
                # Then adjust the last probability to ensure exact sum of 1.0
                if probabilities:
                    last_disease = list(probabilities.keys())[-1]
                    current_sum = sum(probabilities.values())
                    probabilities[last_disease] += (1.0 - current_sum)
        
        # If we have too many, keep only the top num_diseases
        if len(probabilities) > num_diseases:
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            probabilities = {k: v for k, v in sorted_probs[:num_diseases]}
            
            # Normalize one last time
            total = sum(probabilities.values())
            if total > 0:
                probabilities = {k: v/total for k, v in probabilities.items()}
                # Then adjust the last probability to ensure exact sum of 1.0
                last_disease = list(probabilities.keys())[-1]
                current_sum = sum(probabilities.values())
                probabilities[last_disease] += (1.0 - current_sum)
        
        self.previous_probabilities = probabilities
        self.patient_info += additional_info + " "
        
        # If reasoning was requested, return an empty string as reasoning
        if get_reasoning:
            empty_reasoning = "No reasoning provided as reasoning output is disabled."
            return probabilities, empty_reasoning
        return probabilities 