import os
import json
import hashlib
import time
from functools import wraps
from typing import Dict, Any, Callable

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache expiration (1 week in seconds)
CACHE_EXPIRATION = 7 * 24 * 60 * 60

class CachedResponse:
    """Class to mimic OpenAI API response structure"""
    def __init__(self, content: Dict[str, Any]):
        self.choices = [
            type('obj', (object,), {'message': type('obj', (object,), {'content': content.get('content', '')})})
        ]

def create_cache_key(model: str, messages: list) -> str:
    """Create a unique cache key for the API request"""
    # Convert messages to a string and hash it
    if isinstance(messages, list):
        messages_str = json.dumps(messages, sort_keys=True)
    else:
        messages_str = str(messages)
    
    key = f"{model}_{hashlib.md5(messages_str.encode()).hexdigest()}"
    return key

def get_cache_path(cache_key: str) -> str:
    """Get the file path for a cache key"""
    return os.path.join(CACHE_DIR, f"{cache_key}.json")

def save_to_cache(cache_key: str, response: Dict[str, Any]) -> None:
    """Save response to cache file"""
    cache_path = get_cache_path(cache_key)
    cache_data = {
        'timestamp': time.time(),
        'response': response
    }
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f)

def load_from_cache(cache_key: str) -> Dict[str, Any]:
    """Load response from cache file if it exists and is not expired"""
    cache_path = get_cache_path(cache_key)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check if cache is expired
        if time.time() - cache_data['timestamp'] > CACHE_EXPIRATION:
            os.remove(cache_path)  # Remove expired cache
            return None
        
        return cache_data['response']
    except Exception as e:
        print(f"Cache error: {e}")
        return None

def cached_completion(func: Callable) -> Callable:
    """Decorator to cache OpenAI API completions"""
    @wraps(func)
    def wrapper(model, messages, *args, **kwargs):
        # Skip caching for certain scenarios
        if kwargs.get('skip_cache', False):
            kwargs.pop('skip_cache', None)
            return func(model, messages, *args, **kwargs)
        
        # Create cache key
        cache_key = create_cache_key(model, messages)
        
        # Try to load from cache
        cached_response = load_from_cache(cache_key)
        if cached_response:
            print("Using cached response")
            return CachedResponse(cached_response)
        
        # No cache hit, make the actual API call
        response = func(model, messages, *args, **kwargs)
        
        # Save the response to cache
        try:
            response_content = response.choices[0].message.content
            save_to_cache(cache_key, {'content': response_content})
        except (AttributeError, IndexError) as e:
            print(f"Failed to cache response: {e}")
        
        return response
    
    return wrapper

def cached_scenarios(func: Callable) -> Callable:
    """Decorator specifically for scenario probability calculations"""
    @wraps(func)
    def wrapper(self, patient_info, question, number_of_scenarios):
        # Create cache key
        cache_key = f"scenario_{hashlib.md5(patient_info.encode()).hexdigest()}_{hashlib.md5(question.encode()).hexdigest()}"
        cache_path = get_cache_path(cache_key)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache is expired
                if time.time() - cache_data['timestamp'] <= CACHE_EXPIRATION:
                    return cache_data['scenarios']
            except Exception as e:
                print(f"Scenario cache error: {e}")
        
        # No cache hit, calculate scenarios
        scenarios = func(self, patient_info, question, number_of_scenarios)
        
        # Save to cache
        cache_data = {
            'timestamp': time.time(),
            'scenarios': scenarios
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Failed to cache scenarios: {e}")
        
        return scenarios
    
    return wrapper

def cached_entropy(func: Callable) -> Callable:
    """Enhanced entropy caching beyond the basic lru_cache"""
    @wraps(func)
    def wrapper(probabilities_tuple):
        cache_key = f"entropy_{hashlib.md5(str(probabilities_tuple).encode()).hexdigest()}"
        cache_path = get_cache_path(cache_key)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Entropy calculations don't expire
                return cache_data['entropy']
            except Exception:
                pass
        
        # Calculate entropy
        entropy = func(probabilities_tuple)
        
        # Save to cache
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'entropy': entropy}, f)
        except Exception as e:
            print(f"Failed to cache entropy: {e}")
        
        return entropy
    
    return wrapper 