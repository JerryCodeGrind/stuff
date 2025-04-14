# Diagnostic Information Gain Network

This repository contains an implementation of a medical diagnostic system that uses an information gain approach to efficiently diagnose patients by asking optimal questions.

## Key Features

- **Information Gain Optimization**: Automatically selects questions that maximize information gain to efficiently narrow down diagnoses
- **Two Diagnostic Approaches**: 
  - Information Gain Network (IGN): An entropy-based approach that selects optimal questions
  - GPT Doctor: A traditional approach that simulates a doctor asking sequential questions
- **Benchmarking Tools**: Compare the performance of both approaches on diagnostic cases
- **Parallel Processing**: Concurrent evaluation of questions for improved performance
- **Caching System**: API response caching to reduce redundant calls

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)

### Installation

1. Clone the repository
2. Install dependencies: `pip install openai numpy`
3. Set your OpenAI API key: `export OPENAI_API_KEY=your_api_key_here`

### Running the Benchmark

```bash
python run_benchmark.py
```

## How It Works

The Information Gain Network:
1. Starts with a set of potential diagnoses based on initial patient information
2. Calculates the entropy of the current probability distribution
3. For each potential question:
   - Simulates possible patient responses
   - Calculates how each response would update the diagnosis probabilities 
   - Determines the expected information gain
4. Selects the question with the highest information gain
5. Asks the selected question to the patient
6. Updates the diagnosis probabilities based on the response
7. Repeats until confident in a diagnosis or maximum questions reached

## Configuration Parameters

In `run_benchmark.py`:
- `MAX_DISEASES`: Number of diseases to track (default: 10)
- `MAX_QUESTIONS`: Maximum questions to ask (default: 10)
- `MAX_WORKERS`: Number of concurrent threads (default: 4)
- `TURNS_BEFORE_NARROWING`: Questions to ask before narrowing diagnoses (default: 3)
- `CONFIDENCE_THRESHOLD`: Probability threshold for confident diagnosis (default: 0.75)

## License

MIT

## Acknowledgments

- OpenAI for providing the GPT models used in this project 