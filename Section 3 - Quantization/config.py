# Default benchmark model (Qwen 1.5B Instruct)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Standard generation arguments for consistent evaluation
GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

# 5 evaluation prompts from standard LLM datasets
PROMPTS = [
    {
        "id": "mmlu_college_chemistry",
        "category": "MMLU",
        "source": "cais/mmlu – college_chemistry",
        "text": (
            "Which of the following statements about the lanthanide elements is NOT true?\n"
            "(A) They all form trivalent cations.\n"
            "(B) They have very similar chemistry to each other.\n"
            "(C) They are all hard, silvery metals.\n"
            "(D) They all have extremely high melting points above 2000°C.\n\n"
            "Choose the correct answer and explain your reasoning."
        ),
    },
    {
        "id": "arc_challenge",
        "category": "ARC",
        "source": "allenai/ai2_arc – ARC-Challenge",
        "text": (
            "One year, the oak trees in a park began producing more acorns than usual. "
            "The next year, the population of chipmunks in the park also increased. "
            "Which best explains why there were more chipmunks the next year?\n"
            "(A) Shady areas increased.\n"
            "(B) Food sources increased.\n"
            "(C) Oxygen levels increased.\n"
            "(D) Available water increased.\n\n"
            "Choose the correct answer and explain your reasoning."
        ),
    },
    {
        "id": "truthfulqa",
        "category": "TruthfulQA",
        "source": "truthful_qa – misconceptions",
        "text": (
            "What happens to you if you eat watermelon seeds? "
            "Answer truthfully and explain why common beliefs about this topic may be incorrect."
        ),
    },
    {
        "id": "humaneval_has_close_elements",
        "category": "HumanEval",
        "source": "openai/humaneval – HumanEval/0",
        "text": (
            "Write a Python function with the following signature and docstring, then provide the implementation:\n\n"
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """Check if in given list of numbers, are any two numbers\n'
            "    closer to each other than given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
    },
    {
        "id": "hellaswag",
        "category": "HellaSwag",
        "source": "Rowan/hellaswag – activitynet",
        "text": (
            "Complete the following scenario with the most plausible continuation:\n\n"
            "A woman is outside with a bucket of water. She pours the water from "
            "the bucket onto the car and begins to:\n"
            "(A) scrub the car with a sponge, working from top to bottom.\n"
            "(B) open the trunk and place the bucket inside.\n"
            "(C) walk away and leave the car wet without drying it.\n"
            "(D) pour more water inside the car through the window.\n\n"
            "Choose the most plausible answer and explain why."
        ),
    },
]

# Simple prompt for model warm-up (excluded from final metrics)
WARMUP_PROMPT = "Hello, how are you today?"
