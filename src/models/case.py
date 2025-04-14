from typing import Dict, List, TypedDict, Optional

class Case(TypedDict):
    """Represents a patient case with doctor's notes and ground truth diagnosis"""
    doctor_vignette: str
    patient_profile: str
    diagnosis: str

# Define a list of patient cases
cases = [
  {
    "doctor_vignette": "\"A 28-year-old female school teacher presents with a six-month history of multiple, increasing, rough dermatological lesions on her hands, feet, and elbows, associated with occasional itching and pain when walking. The lesions are unresponsive to over-the-counter treatments and are causing her embarrassment and social anxiety, leading her to seek dermatological advice.\"",
    "patient_profile": "You are a 28-year-old woman who works as a school teacher and lives alone in a small apartment. Over the past six months, you have developed multiple raised and rough dermatological lesions on your hands, feet, and around your elbows. Initially, you thought they were harmless, perhaps just calluses from frequent writing and working with students, but they have become more numerous and bothersome. The lesions on your hands occasionally itch, but most disturbingly, they're becoming quite embarrassing, making you self-conscious when interacting with colleagues and students. \n\nYou recall that your younger brother had similar skin issues when he was a child, and a doctor had mentioned they were warts caused by the human papillomavirus (HPV). You've tried over-the-counter wart treatments, but nothing seems to work; in fact, some lesions appear to be growing rather than diminishing. You've also noticed some warts developing on the soles of your feet, which cause discomfort and pain when walking.\n\nYour medical history includes mild asthma, which is well-controlled with an albuterol inhaler. You have no history of skin problems besides the current issue, and you are generally healthy. You live an active lifestyle and enjoy running on weekends. You are up to date on your vaccinations, including the HPV vaccine, and you hardly ever drink alcohol, usually just on social occasions.\n\nIn your personal life, you have been feeling a bit anxious about how the warts could affect your social and dating life. You're concerned they could be contagious, and you've avoided wearing sandals and swimming in public pools. Recently, a friend suggested seeing a dermatologist, so you've made an appointment, hoping they can find an effective treatment to help you feel more comfortable in your own skin again. Your biggest fear is that these warts might indicate a more serious underlying health issue, but you're also just eager to get them removed so you can gain back your confidence and feel normal again.",
    "diagnosis": "Warts"
  }
]

# Added potential common skin condition diagnoses to ensure at least 5 diseases are included
POTENTIAL_SKIN_DIAGNOSES = [
    "Warts",
    "Psoriasis",
    "Eczema (Atopic Dermatitis)",
    "Contact Dermatitis",
    "Seborrheic Keratosis",
    "Actinic Keratosis",
    "Lichen Planus",
    "Folliculitis",
    "Basal Cell Carcinoma",
    "Squamous Cell Carcinoma"
]

class DiagnosisResult(TypedDict):
    """Represents the results of a diagnostic run"""
    questions_asked: int
    diagnoses: List[Dict]
    confident_diagnosis: bool
    correct_diagnosis: bool
    final_diagnosis: Optional[str]
    final_probability: float
    ground_truth: str
    narrowing_events: List[Dict]
    final_disease_count: Optional[int]
    narrowing_steps: Optional[int] 