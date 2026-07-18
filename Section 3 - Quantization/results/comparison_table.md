# Quantization Benchmark Results

**Model:** Qwen/Qwen2.5-1.5B-Instruct  
**Date:** 2026-07-19 01:13  
**Hardware:** NVIDIA RTX 3060 6GB (Laptop)

## Trade-off Summary: Precision vs. Size vs. Speed vs. Quality

| Dimension         | FP16 (Full Precision)    | NF4 4-bit (Quantized)                       |
|-------------------|--------------------------|---------------------------------------------|
| Precision         | 16-bit floating point    | 4-bit NormalFloat (NF4)                     |
| Model Size (VRAM) | 2944 MB                  | 1109 MB                                     |
| Param Memory      | 2944 MB                  | 1070 MB                                     |
| Speed (tok/s)     | 17.9                     | 9.5                                         |
| Quality           | Baseline (full fidelity) | Near-baseline (see output comparison below) |

## Detailed Performance & Memory Comparison

| Metric | FP16 (baseline) | NF4 4-bit | Delta |
|--------|-----------------|-----------|-------|
| GPU VRAM (MB) | 2944.4 | 1109.2 | +62.3% saved |
| Param Memory (MB) | 2944.4 | 1070.2 | +63.7% saved |
| RAM / RSS (MB) | 1216.2 | 1134.8 | - |
| Avg Tokens/sec | 17.9 | 9.5 | -46.8% |

## Per-Prompt Throughput (tokens/sec)

| Prompt     |   FP16 tok/s |   FP16 tokens |   FP16 time (s) |   NF4 tok/s |   NF4 tokens |   NF4 time (s) |
|------------|--------------|---------------|-----------------|-------------|--------------|----------------|
| MMLU       |         18.3 |           210 |           11.48 |         9.9 |          146 |          14.77 |
| ARC        |         18.7 |           234 |           12.51 |        10   |           97 |           9.68 |
| TruthfulQA |         17.2 |           247 |           14.35 |         9.7 |          190 |          19.7  |
| HumanEval  |         17   |           256 |           15.05 |        10   |          256 |          25.69 |
| HellaSwag  |         18.1 |           221 |           12.23 |         8   |          256 |          32.02 |

## Qualitative Output Comparison

*Same 5 prompts run through both FP16 and NF4 versions for side-by-side quality assessment.*

### MMLU

**Prompt:** Which of the following statements about the lanthanide elements is NOT true?
(A) They all form trivalent cations.
(B) They have very similar chemistry to each other.
(C) They are all hard, silvery metals.
(D) They all have extremely high melting points above 2000°C.

Choose the correct answer and explain your reasoning.

**FP16 Output:**
```
The statement that is not true among the options given is:

(C) They are all hard, silvery metals.

Explanation: The lanthanides (elements with atomic numbers from 57 to 71) do indeed share some properties due to their similar chemical behavior, but they are not generally described as "hard, silvery metals." Here’s why this statement is false:

- **Hardness**: Lanthanides typically exhibit metallic characteristics rather than being rigid or hard like certain transition metals. Their hardness can vary depending on specific compounds formed.

- **Silvery Appearance**: While many lanthanides have metallic luster when freshly cut, the term "silvery" is often used more broadly in metallurgy to describe silver-colored alloys rather than pure elemental forms.

So, while lanthanides share common physical and chemical behaviors due to having nearly identical electron configurations in the f-orbital shells, describing them as "hard, silvery metals" does not fully capture the diversity and range of their properties across different compounds and environments.
```

**NF4 4-bit Output:**
```
The statement that is NOT true about the lanthanide elements is:

(A) They all form trivalent cations.

Explanation: The lanthanides do indeed form trivalent cations (La³⁺, Ce³⁺, etc.), but they also form divalent cations when in compounds with certain transition metal ions or halogen anions. For example, the oxide ion (O²⁻), which can bind to different lanthanides to form various oxidation states such as +3, +4, +5, -2, etc., not just +3 like in most divalent compounds. This makes option A incorrect because it suggests uniform trivalency without accounting for potential higher valence states in some complexes.
```

---

### ARC

**Prompt:** One year, the oak trees in a park began producing more acorns than usual. The next year, the population of chipmunks in the park also increased. Which best explains why there were more chipmunks the next year?
(A) Shady areas increased.
(B) Food sources increased.
(C) Oxygen levels increased.
(D) Available water increased.

Choose the correct answer and explain your reasoning.

**FP16 Output:**
```
The correct answer is (B) Food sources increased.

Explanation: Oak trees produce acorns as a food source for animals like squirrels and chipmunks. When an oak tree begins to produce more acorns than usual, it means that there is more available food for these animals. This increase in food availability would naturally lead to an increase in the population of chipmunks in the following year. 

Let's analyze each option:
- (A) Shady areas increased: While shade can affect plant growth and habitat conditions, it does not directly cause an increase in acorn production or chipmunk populations.
- (C) Oxygen levels increased: Changes in oxygen levels do not directly impact the production or consumption of acorns by animals.
- (D) Available water increased: Similar to changes in oxygen levels, an increase in water availability doesn't directly influence the number of acorns produced by oak trees or their consumption by animals.

Therefore, the most logical explanation for the increase in chipmunk population over the next year is that the food source (acorns from oak trees) had increased due to the unusual production of acorns by the oak trees.
```

**NF4 4-bit Output:**
```
The correct answer is (B) Food sources increased.

Reasoning: The increase in oak tree production led to an abundance of acorns that became available as food resources for chipmunks. This directly increases the availability of food sources for the chipmunks, which can lead to an increase in their population size. The other options (shading, oxygen levels, and water) do not have a direct impact on the number of chipmunk populations based on the information provided.
```

---

### TruthfulQA

**Prompt:** What happens to you if you eat watermelon seeds? Answer truthfully and explain why common beliefs about this topic may be incorrect.

**FP16 Output:**
```
When you eat watermelon seeds, they will pass through your digestive system without causing harm or leading to any specific health issues. The belief that eating watermelon seeds can cause kidney stones is not accurate.

Watermelon seeds contain oxalates, which can contribute to the formation of kidney stones when consumed in excess. However, most people do not need to worry about this because their bodies naturally produce enough oxalates to prevent stone formation. Additionally, it's important to note that while some individuals may have an increased risk for kidney stones due to certain genetic factors or other medical conditions, consuming watermelon seeds does not significantly increase this risk unless someone has been diagnosed with kidney stones before.

It's also worth noting that watermelons themselves do not cause kidney stones; rather, it's the consumption of large quantities of watermelon seeds (or any other food high in oxalates) that might lead to stone formation.

In summary, while watermelon seeds contain oxalates, the amount typically found in a serving size makes them unlikely to cause kidney stones. If you're concerned about kidney stone formation, it would generally be more prudent to focus on overall dietary habits and individual health assessments rather than worrying solely about watermelon seed intake.
```

**NF4 4-bit Output:**
```
If someone eats the seeds of a watermelon, they will not become watermelons because they do not contain any chlorophyll or photosynthetic pigments that are essential for plant growth. Watermelons grow from fruits called "seedless" watermelons, which are actually modified tomato flowers. The seeds in these fruits are not the same as those found in regular seed-bearing plants.

However, eating the seeds is harmless but can cause digestive issues for some people due to their hard texture and high levels of oxalate and phytic acid. These compounds can interfere with mineral absorption in the intestines. Additionally, consuming them might lead to constipation if swallowed whole.

Common beliefs suggesting that ingesting the seeds would somehow result in becoming a watermelon could stem from the idea that seeds have magical properties in mythology and folklore. In reality, there's no scientific basis for such claims regarding seeds' effects on one's body or identity.
```

---

### HumanEval

**Prompt:** Write a Python function with the following signature and docstring, then provide the implementation:

from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers
    closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """


**FP16 Output:**
```
```python
from typing import *
from collections import defaultdict

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    Check if in given list of numbers, are any two numbers closer to each other 
    than given threshold.

    Args:
        numbers (List[float]): The list of floating-point numbers.
        threshold (float): The maximum allowed difference between any two numbers.

    Returns:
        bool: True if there exist at least two numbers that satisfy the condition,
              False otherwise.
    """

    # Create a dictionary to store the last occurrence index of each number
    last_occurrence = defaultdict(int)

    for i, num in enumerate(numbers):
        # Calculate the absolute difference between the current number and all previously seen numbers
        differences = [abs(num - prev_num) for prev_num in numbers[:i]]
        
        # If the minimum difference is less than or equal to the threshold, return True
        if min(differences) <= threshold:
            return True
        
        # Update the last occurrence index of the current number
        last_occurrence[num] = i
    
    # If no such pair was found, return False
    return False


# Test cases
print(has_close_elements([1.0,
```

**NF4 4-bit Output:**
```
```python
from typing import List
import math

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    Check if in given list of numbers, are any two numbers
    closer to each other than given threshold.
    
    Args:
        numbers (List[float]): A list of floating-point numbers.
        threshold (float): The threshold value for comparing distances between numbers.

    Returns:
        bool: True if there exists at least one pair of numbers whose distance is less than the threshold, otherwise False.
    """

    # Sort the numbers first
    sorted_numbers = sorted(numbers)
    
    # Iterate through the sorted list to find pairs that meet the condition
    for i in range(len(sorted_numbers)):
        if i > 0:
            current_distance = sorted_numbers[i] - sorted_numbers[i-1]
            if current_distance < threshold:
                return True
    
    return False

# Test cases
print(has_close_elements([1.0, 2.0, 3.0], 0.5))  # Expected output: False
print(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0
```

---

### HellaSwag

**Prompt:** Complete the following scenario with the most plausible continuation:

A woman is outside with a bucket of water. She pours the water from the bucket onto the car and begins to:
(A) scrub the car with a sponge, working from top to bottom.
(B) open the trunk and place the bucket inside.
(C) walk away and leave the car wet without drying it.
(D) pour more water inside the car through the window.

Choose the most plausible answer and explain why.

**FP16 Output:**
```
The most plausible answer is (A) scrub the car with a sponge, working from top to bottom.

This action makes the most sense in the given context because the goal appears to be cleaning or washing the car. Scrubbing the car with a sponge would involve applying pressure to remove dirt, grime, and other contaminants from the vehicle's surface. This method aligns well with the idea of "pouring the water" and then proceeding to clean the car.

Option (B) seems illogical as opening the trunk doesn't help with cleaning the exterior of the car directly unless you're adding more water to the bucket already present on the ground.

Option (C) contradicts the initial action of pouring water and suggests leaving the car unattended, which isn't an efficient way to use resources like water and time.

Option (D) also doesn’t make practical sense, as pouring more water into the car wouldn't typically contribute to cleaning if you're trying to wash the exterior.

Therefore, option A provides the most logical next step in this sequence of actions for cleaning a car using water.
```

**NF4 4-bit Output:**
```
The most plausible answer is (A) scrub the car with a sponge, working from top to bottom.

This option involves cleaning the car's exterior thoroughly, which is an appropriate use for the water that has been poured onto the vehicle. Scrubbing the car can help remove dirt, grime, and other contaminants on the surface of the paint or metal parts. Starting from top to bottom ensures even coverage and prevents missed spots in areas like roof lines and door seams. This method also helps distribute the water effectively across different surfaces of the vehicle, ensuring thorough cleaning.

Options (B), (C), and (D) do not align well with the practical implications of pouring water directly onto a car:

- Option (B) suggests placing the bucket inside the trunk, but this would be inefficient as the bucket itself needs to be cleaned before being used again.
- Option (C) implies leaving the car wet without drying it, which could lead to potential damage if water remains in contact with sensitive components such as electronics or battery systems.
- Option (D) introduces the idea of pouring more water into the car via a window, which contradicts the initial action of removing water from the bucket.

Therefore, option (A) provides the most logical and practical continuation of the described scenario.
```

---

