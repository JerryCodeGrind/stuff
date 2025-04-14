import concurrent.futures
from typing import List, Dict
from src.utils.api import call_completion_api

class ProbabilityAgent:
    def __init__(self, questions_per_disease: int = 2):
        """
        Initialize the Probability Agent.
        
        Args:
            questions_per_disease: Number of questions to generate for each disease category
        """
        self.questions_per_disease = questions_per_disease
        self.question_set = self._generate_question_set()
    
    def _generate_question_set(self) -> List[str]:
        """
        Generate a set of diagnostic questions.
        
        This will create questions_per_disease questions for each of several common disease categories.
        The categories cover major body systems and types of diseases.
        """
        # Define common disease categories/body systems to ensure good coverage
        disease_categories = [
            "Cardiovascular (heart failure, coronary artery disease, arrhythmias)",
            "Respiratory (pneumonia, asthma, COPD, pulmonary embolism)",
            "Gastrointestinal (gastritis, peptic ulcer, hepatitis, pancreatitis)",
            "Neurological (stroke, seizures, migraines, multiple sclerosis)",
            "Endocrine (diabetes, thyroid disorders, adrenal disorders)",
            "Infectious (bacterial infections, viral infections, fungal infections)",
            "Rheumatological (arthritis, lupus, fibromyalgia)",
            "Psychiatric (depression, anxiety, bipolar disorder)",
            "Oncological (various cancers, paraneoplastic syndromes)",
            "Renal (kidney failure, kidney stones, urinary tract infections)"
        ]
        
        # Calculate how many categories we need based on the requested questions_per_disease
        # to ensure we get diverse coverage of different medical areas
        num_categories = min(len(disease_categories), 10)  # Use at most 10 categories
        
        # Generate questions for each category in parallel
        all_questions = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit tasks
            future_to_category = {
                executor.submit(self._generate_category_questions, disease_categories[i]): i 
                for i in range(num_categories)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_category):
                category_questions = future.result()
                all_questions.extend(category_questions)
        
        return all_questions
    
    def _generate_category_questions(self, category: str) -> List[str]:
        """Generate questions for a specific disease category"""
        prompt = f"""Generate exactly {self.questions_per_disease} specific diagnostic questions related to {category}.
        Each question should help differentiate between different conditions within this category.
        Questions should be highly informative for diagnosis and should have high discriminative value.
        Return ONLY the questions, one per line, with no additional text or numbering.
        
        Example format for {category}:
        [Specific question about symptoms related to this category]
        [Specific question about risk factors related to this category]
        [Specific question about duration or pattern of symptoms related to this category]
        """
        
        response = call_completion_api(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response to get just the questions
        category_questions = [
            q.strip() for q in response.choices[0].message.content.split('\n')
            if q.strip() and not q.startswith(('Example', 'Format', '[', '-', '*'))
        ]
        
        # Limit to exactly questions_per_disease questions per category
        return category_questions[:self.questions_per_disease]
    
    def calculate_scenario_probabilities(self, patient_info: str, question: str, number_of_scenarios: int) -> Dict[str, float]:
        prompt = f"""Based on this patient information:
        {patient_info}
        
        For this specific question: {question}
        
        Generate {number_of_scenarios} distinct possible patient responses and their probabilities.
        Each line should be in the format: response|probability
        Probabilities should be numbers between 0 and 1 (to 3 decimal places).
        The sum of all probabilities must equal 1.0.
        
        Guidelines:
        1. Responses should be distinct and not overlap in meaning
        2. Probabilities should reflect realistic likelihoods
        3. Include a mix of clear yes/no responses and nuanced responses
        4. Do not include duplicate or similar responses
        
        Example format:
        Yes, I experience severe pain|0.300
        Yes, but only mild discomfort|0.250
        No, I don't experience any pain|0.200
        Sometimes, depending on activity|0.150
        I'm not sure|0.100
        """
        
        response = call_completion_api(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        scenarios = {}
        seen_responses = set()
        for line in response.choices[0].message.content.split('\n'):
            if '|' in line:
                scenario, prob = line.split('|')
                scenario = scenario.strip()
                if scenario not in seen_responses:  # Prevent duplicates
                    scenarios[scenario] = float(prob.strip())
                    seen_responses.add(scenario)
        
        return scenarios 