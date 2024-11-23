# Episodic Memory Enhanced Language Agent

## Overview

This project explores the integration of an **episodic memory module** into a language agent designed for sequential learning tasks that require long-term retention and generalization. The goal to demonstrate how episodic memory improves performance in handling ambiguous, rare, novel, and interference-prone linguistic patterns.

The language agent is built using **PyACTR**, a Python implementation of the ACT-R cognitive architecture, which models human cognitive processes such as memory, attention, and problem-solving. **PyACTR** provides the foundational cognitive processes and memory systems necessary for simulating human-like language processing. 
The experiment involves a series of carefully designed test cases across linguistic categories and patterns, comparing the agent's performance **with** and **without** episodic memory (EM). 


---

## Key Features

1. **Episodic Memory Module**:
   - Tracks and retrieves context-specific episodes to resolve ambiguities, handle rare cases, and generalize from past experiences.
   - Implements context vector enrichment with temporal, performance, and recency-based features.

2. **Test Case Diversity**:
   - Covers multiple linguistic phenomena, including:
     - Regular patterns (e.g., pluralization rules).
     - Ambiguous homonyms (e.g., "lead" as guidance vs. material).
     - Rare irregular forms (e.g., "criterion" → "criteria").
     - Novel nonsense words (e.g., "florp" → "florps").
     - Suppletive forms (e.g., "go" → "went").
     - Extreme edge cases with high contextual conflicts (e.g., "set" in math vs. collections).

3. **Performance Metrics**:
   - Success rate and processing time are measured across categories.
   - Visual comparisons (graphs included) to highlight improvements in efficiency and adaptability with episodic memory.

4. **Visualization and Analysis**:
   - Detailed plots for success rate, processing time, and performance by category/pattern.
   - Learning curve comparisons to showcase the benefits of episodic memory.

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher.
- Required libraries (install via `pip`):
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

### Clone the Repository

```bash
git clone https://github.com/Yaima/episodic-memory-pyactr.git
cd episodic-memory-pyactr
```

### Install Dependencies

```bash
pip install -r requirements.txt
```
## Usage

### Run Experiments

To execute the experiment with and without episodic memory:

```bash
python main.py
```

## Output Details

### Experiment Results
- Printed to the console for overall metrics, performance by category, and detailed test case results.

### Logs and Visualizations

#### Logs:
- `outputs/logs/experiment_log.txt`
- `outputs/logs/llm_analysis.md`

#### Plots:
- `outputs/plots/comparison.png` (overall metrics and processing time analysis).
- `outputs/plots/learning_curve.png` (processing time vs. case number).
- `outputs/plots/frequency_analysis.png` (processing time by word frequency).
- `outputs/plots/efficiency_comparison.png` (efficiency comparison).
- `outputs/plots/memory_utilization.png` (memory utilization).

---

## Customize Test Cases

Test cases can be modified or extended in `main.py` under the `generate_test_cases` function. Adjust these to explore different linguistic challenges or patterns.

### Suggested Parameters
- The following configuration yielded the best balance between accuracy and processing time:
  - `retrieval_threshold`: 0.35
  - `decay`: 0.7
  - `instantaneous_noise`: 0.5
  - `partial_matching`: False

---

## Results

### Overall Performance:
- **100% success rate** across all test cases, both with and without episodic memory.
- Episodic memory consistently reduces **average processing time** (e.g., 0.0054s vs. 0.0096s in the provided experiment).

### Learning Efficiency:
- Episodic memory adapts more efficiently to novel and interference-prone cases, as shown in the learning curve.
- Handling ambiguous homonyms and rare irregular forms becomes significantly faster with episodic memory.

### Scalability:
- Episodic memory demonstrates scalability, maintaining low processing time and high accuracy across 100 stored episodes. Testing with larger datasets could identify performance thresholds and guide memory optimization strategies.

### Partial Matching Impact:
- Enabling partial matching introduced inaccuracies in some cases, especially for ambiguous and interference-prone patterns. 
- Disabling partial matching significantly enhances accuracy while maintaining efficiency, indicating that precise matching is more beneficial for episodic memory modules in handling ambiguous cases.


| Configuration            | Success Rate | Avg Processing Time (s) |
|--------------------------|--------------|-------------------------|
| With Partial Matching    | 45-50%       | 0.012                   |
| Without Partial Matching | 100%         | 0.005                   |



---

## Visualizations

Generated plots include:

1. **Overall Agreement Success Rate**:
   - Success Rate: Uniform 100% success across methods, demonstrating consistent system reliability.
   - Processing Time Distribution: Episodic memory reduces average processing time significantly compared to non-episodic methods.
   - Learning Curve: Shows efficiency gains in handling sequential tasks over time.
   - Processing Time by Word Frequency and Pattern: Highlights episodic memory’s effectiveness with low-frequency and complex patterns.
2. **Processing Time Distribution**:
   - Visualizes the trade-offs introduced by enabling partial matching, helping to identify optimal configurations for specific use cases.
3. **Performance by Category**:
   - Confirms robust performance across regular, ambiguous, irregular, novel, suppletive, interference, and extreme categories.
4. **Learning Curve**:
   - Clear efficiency gains in handling sequential tasks with episodic memory.
5. **Processing Time by Word Frequency and Pattern**:
   - Shows episodic memory's benefits in handling low-frequency and complex patterns.

---

## Research Contribution

This project addresses the question:

> **How does the integration of an episodic memory module into the ACT-R cognitive architecture affect processing efficiency and accuracy in linguistic tasks involving context-dependent meaning resolution?**

### Key Insights:
- **Architectural Advantage:** The integration of episodic memory provides a complementary mechanism to ACT-R's traditional memory systems, enabling more efficient context-sensitive processing without compromising accuracy.

- **Retrieval Strategy:** First-attempt retrieval patterns don't need to be perfect (as low as 8% for complex cases) as long as the system has robust backup processing mechanisms, demonstrating the value of a layered memory approach.

- **Scalability Pattern:** Linear memory growth with consistent performance suggests this approach could be viable for larger-scale applications, though memory optimization strategies would be needed.


---

## Future Directions

### Expand Test Cases:
- Introduce multilingual test cases and morphologically richer languages.
- Simulate real-world linguistic challenges like idiomatic expressions.

### Memory Optimization:
- Investigate strategies for dynamic memory cleanup in large-scale systems.

### Integration with Larger Architectures:
- Combine episodic memory with reinforcement learning for more adaptive decision-making.

### Automated LLM Feedback:
- Automate analysis with advanced language models for detailed analysis and iterative improvement of episodic memory modules.

#### Environment Integration:
- Investigate the use of interactive environments to simulate dynamic agent-environment interactions and evaluate episodic memory's performance in non-static scenarios.

---

## Contact

For questions, contributions, or feedback, please reach out via:

- **GitHub**: [https://github.com/Yaima](https://github.com/Yaima)
- **Issues**: [Submit feedback or contributions here](https://github.com/Yaima/episodic-memory-pyactr/issues)

Alternatively, you can email: **[yaimamvaldivia@gmail.com](mailto:yaimamvaldivia@gmail.com)**

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



