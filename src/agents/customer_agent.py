from src.utils.api import call_completion_api

class CustomerAgent:
    def __init__(self, patient_profile: str):
        self.patient_profile = patient_profile
    
    def respond_to_question(self, question: str) -> str:
        prompt = f"""You are a patient with the following medical history and symptoms:
        {self.patient_profile}
        
        The doctor asks: {question}
        
        Act as if you are a regular patient. Do not reveal any information about your diagnosis or anything related to your diagnosis. Do not respond with an extremely long answer

        Respond with a concise answer. Try to make some answers ambiguous as you are just a patient and aren't sure of what your diagnosis is.
        
        Include relevant details about your symptoms and be specific about your experience.
        If you're unsure, indicate your uncertainty.
        
        Make sure your response reflects how an average patient would respond to the question. Short and concise.
        
        """
        
        response = call_completion_api(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.choices[0].message.content.strip()
        return answer 