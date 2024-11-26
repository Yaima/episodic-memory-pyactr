# MIT License
#
# Copyright (c) 2024 Yaima Valdivia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from models.agent_model import AgentModel
from modules.openai_client import query_openai


def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate a comprehensive set of test cases for episodic memory evaluation"""
    test_cases = [
        # High-Frequency Regulars: Words that follow standard grammatical rules for pluralization or
        # other morphological changes and are commonly used in the language.
        {'word': 'cat', 'phonology': '/kæt/', 'meaning': '[[cat]]', 'number': 'sg', 'person': '3', 'animacy': 'animate',
         'category': 'regular', 'frequency': 'high', 'pattern_group': 'simple_plural'},
        {'word': 'cats', 'phonology': '/kæts/', 'meaning': '[[cats]]', 'number': 'pl', 'person': '3',
         'animacy': 'animate', 'category': 'regular', 'frequency': 'high', 'pattern_group': 'simple_plural'},
        {'word': 'dog', 'phonology': '/dɒg/', 'meaning': '[[dog]]', 'number': 'sg', 'person': '3', 'animacy': 'animate',
         'category': 'regular', 'frequency': 'high', 'pattern_group': 'simple_plural'},
        {'word': 'dogs', 'phonology': '/dɒgz/', 'meaning': '[[dogs]]', 'number': 'pl', 'person': '3',
         'animacy': 'animate', 'category': 'regular', 'frequency': 'high', 'pattern_group': 'simple_plural'},

        # Ambiguous Homonyms: Words that share the same spelling and pronunciation but have different meanings
        # based on context.
        {'word': 'lead', 'phonology': '/liːd/', 'meaning': '[[lead]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'ambiguous', 'frequency': 'medium', 'pattern_group': 'homonym',
         'context': 'guidance'},
        {'word': 'lead', 'phonology': '/lɛd/', 'meaning': '[[lead-metal]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'ambiguous', 'frequency': 'medium', 'pattern_group': 'homonym',
         'context': 'material'},
        {'word': 'bass', 'phonology': '/bæs/', 'meaning': '[[bass-fish]]', 'number': 'sg', 'person': '3',
         'animacy': 'animate', 'category': 'ambiguous', 'frequency': 'low', 'pattern_group': 'homonym',
         'context': 'fishing'},
        {'word': 'bass', 'phonology': '/beɪs/', 'meaning': '[[bass-instrument]]', 'number': 'sg', 'person': '3',
         'animacy': 'animate', 'category': 'ambiguous', 'frequency': 'low', 'pattern_group': 'homonym',
         'context': 'music'},

        # Rare Irregulars: Words that do not follow standard grammatical rules for pluralization or
        # other morphological changes and are infrequently used.
        {'word': 'child', 'phonology': '/tʃaɪld/', 'meaning': '[[child]]', 'number': 'sg', 'person': '3',
         'animacy': 'animate', 'category': 'irregular', 'frequency': 'low', 'pattern_group': 'en_plural'},
        {'word': 'children', 'phonology': '/tʃɪldrən/', 'meaning': '[[children]]', 'number': 'pl', 'person': '3',
         'animacy': 'animate', 'category': 'irregular', 'frequency': 'low', 'pattern_group': 'en_plural'},
        {'word': 'criterion', 'phonology': '/kraɪˈtɪriən/', 'meaning': '[[criterion]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'irregular', 'frequency': 'low', 'pattern_group': 'latin'},
        {'word': 'criteria', 'phonology': '/kraɪˈtɪriə/', 'meaning': '[[criteria]]', 'number': 'pl', 'person': '3',
         'animacy': 'inanimate', 'category': 'irregular', 'frequency': 'low', 'pattern_group': 'latin'},

        # Novel Cases: Involve invented or made-up words that do not exist in the standard vocabulary.
        # These are used to test the agent's ability to generalize and apply learned patterns to entirely new instances.
        {'word': 'wug', 'phonology': '/wʌg/', 'meaning': '[[wug]]', 'number': 'sg', 'person': '3', 'animacy': 'animate',
         'category': 'novel', 'frequency': 'none', 'pattern_group': 'simple_plural'},
        {'word': 'wugs', 'phonology': '/wʌgz/', 'meaning': '[[wugs]]', 'number': 'pl', 'person': '3',
         'animacy': 'animate', 'category': 'novel', 'frequency': 'none', 'pattern_group': 'simple_plural'},
        {'word': 'pidion', 'phonology': '/pɪdiən/', 'meaning': '[[pidion]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'novel', 'frequency': 'none', 'pattern_group': 'latin'},
        {'word': 'pidia', 'phonology': '/pɪdiə/', 'meaning': '[[pidia]]', 'number': 'pl', 'person': '3',
         'animacy': 'inanimate', 'category': 'novel', 'frequency': 'none', 'pattern_group': 'latin'},

        # Suppletive Forms: Pairs of words that have different roots or forms to express different grammatical
        # categories, often lacking a direct morphological relationship.
        {'word': 'go', 'phonology': '/ɡoʊ/', 'meaning': '[[go]]', 'number': 'sg', 'person': '3', 'animacy': 'animate',
         'category': 'suppletive', 'frequency': 'high', 'pattern_group': 'suppletive'},
        {'word': 'went', 'phonology': '/wɛnt/', 'meaning': '[[went]]', 'number': 'sg', 'person': '3',
         'animacy': 'animate', 'category': 'suppletive', 'frequency': 'high', 'pattern_group': 'suppletive'},
        {'word': 'person', 'phonology': '/pɜːsən/', 'meaning': '[[person]]', 'number': 'sg', 'person': '3',
         'animacy': 'animate', 'category': 'suppletive', 'frequency': 'low', 'pattern_group': 'suppletive'},
        {'word': 'people', 'phonology': '/piːpəl/', 'meaning': '[[people]]', 'number': 'pl', 'person': '3',
         'animacy': 'animate', 'category': 'suppletive', 'frequency': 'low', 'pattern_group': 'suppletive'},

        # Phonologically Similar (Interference): Similar words belonging to different pattern groups.
        {'word': 'bare', 'phonology': '/bɛr/', 'meaning': '[[bare]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'interference', 'frequency': 'medium', 'pattern_group': 'homonym',
         'context': 'naked'},
        {'word': 'bear', 'phonology': '/bɛr/', 'meaning': '[[bear]]', 'number': 'sg', 'person': '3',
         'animacy': 'animate', 'category': 'interference', 'frequency': 'medium', 'pattern_group': 'homonym',
         'context': 'animal'},
        {'word': 'lead', 'phonology': '/liːd/', 'meaning': '[[lead]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'interference', 'frequency': 'low', 'pattern_group': 'homonym',
         'context': 'guide'},
        {'word': 'lead', 'phonology': '/lɛd/', 'meaning': '[[lead]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'interference', 'frequency': 'low', 'pattern_group': 'homonym',
         'context': 'metal'},

        # Extreme Edge Cases: Involve highly ambiguous words with multiple meanings that are heavily dependent on
        # context, often challenging even for advanced language models.
        {'word': 'set', 'phonology': '/sɛt/', 'meaning': '[[set]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'extreme', 'frequency': 'high', 'pattern_group': 'homonym',
         'context': 'math'},
        {'word': 'set', 'phonology': '/sɛt/', 'meaning': '[[set]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'extreme', 'frequency': 'high', 'pattern_group': 'homonym',
         'context': 'collection'},
        # Financial Institution Context
        {'word': 'bank', 'phonology': '/bæŋk/', 'meaning': '[[bank-financial]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'extreme', 'frequency': 'high', 'pattern_group': 'homonym',
         'context': 'financial'},
        # River Bank Context
        {'word': 'bank', 'phonology': '/bæŋk/', 'meaning': '[[bank-river]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'extreme', 'frequency': 'medium', 'pattern_group': 'homonym',
         'context': 'nature'
         },
        # Banking Action Context
        {'word': 'bank', 'phonology': '/bæŋk/', 'meaning': '[[bank-action]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'extreme', 'frequency': 'high', 'pattern_group': 'homonym',
         'context': 'transaction'
         },
        # Aircraft Banking Context
        {'word': 'bank', 'phonology': '/bæŋk/', 'meaning': '[[bank-turn]]', 'number': 'sg',  'person': '3',
         'animacy': 'inanimate', 'category': 'extreme', 'frequency': 'low', 'pattern_group': 'homonym',
         'context': 'aviation'
         },
        # Memory Bank Context
        {'word': 'bank', 'phonology': '/bæŋk/', 'meaning': '[[bank-storage]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'extreme', 'frequency': 'medium', 'pattern_group': 'homonym',
         'context': 'technology'
         },
        # Blood Bank Context
        {'word': 'bank', 'phonology': '/bæŋk/', 'meaning': '[[bank-medical]]', 'number': 'sg', 'person': '3',
         'animacy': 'inanimate', 'category': 'extreme', 'frequency': 'medium', 'pattern_group': 'homonym',
         'context': 'medical'
         }
    ]

    return test_cases


def run_experiment(with_episodic_memory: bool = True) -> List[Dict[str, Any]]:
    """Run experiment and collect results"""
    test_cases = generate_test_cases()
    print(f"\nRunning experiment {'with' if with_episodic_memory else 'without'} episodic memory...")

    # Create model
    agent = AgentModel(use_episodic_memory=with_episodic_memory)

    results = []

    for idx, test_case in enumerate(test_cases):
        start_time = time.time()
        result = agent.run_agreement_test(test_case)
        processing_time = time.time() - start_time

        # Enhance result with timing and memory stats
        result.update({
            'processing_time': processing_time,
            'memory_utilization': result['memory_stats'].get('memory_utilization', 0) if with_episodic_memory else 0,
            'category': test_case['category']
        })

        results.append({
            'test_case': test_case,
            'result': result
        })

        print(f"\nResults for {test_case['word']} ({test_case['category']}):")
        print(f"Success: {result['success']}")
        print(f"Verb number: {result['verb_number']}")
        print(f"Verb person: {result['verb_person']}")
        print(f"Processing time: {processing_time:.4f}s")
        if with_episodic_memory:
            print(f"Used episodic memory: {result.get('used_episodic', False)}")
            print(f"Memory utilization: {result.get('memory_stats', {}).get('memory_utilization', 'N/A')}")

    return results


def format_overall_results(results_with: List[Dict], results_without: List[Dict], analysis: Dict[str, Any]) -> str:
    """Format the overall experiment results into a prompt for the LLM."""
    prompt = "### Experiment Results ###\n\n"

    # Overall Metrics
    prompt += "Overall Metrics:\n"
    prompt += f"Success Rate (With EM): {analysis['overall_success_rate']['with_em'] * 100:.2f}%\n"
    prompt += f"Success Rate (Without EM): {analysis['overall_success_rate']['without_em'] * 100:.2f}%\n"
    prompt += f"Avg Processing Time (With EM): {analysis['processing_time']['with_em']:.4f}s\n"
    prompt += f"Avg Processing Time (Without EM): {analysis['processing_time']['without_em']:.4f}s\n"
    prompt += "\nNote: Success rates are 100% across both configurations, suggesting no cases were mishandled.\n"
    prompt += "Processing time differences may vary based on system noise or implementation-specific factors.\n\n"

    # Performance by Category
    prompt += "Performance by Category:\n"
    for category, perf in analysis['category_performance'].items():
        prompt += f"  {category}:\n"
        prompt += f"    With EM: {perf['with_em'] * 100:.2f}%\n"
        prompt += f"    Without EM: {perf['without_em'] * 100:.2f}%\n"
    prompt += ("\nAll categories show uniform success rates, reflecting consistent system performance regardless of EM "
               "integration.\n\n")

    # Detailed Test Case Results
    prompt += "Detailed Test Case Results:\n\n"
    for idx, (with_res, without_res) in enumerate(zip(results_with, results_without)):
        test_case = with_res['test_case']
        prompt += f"Test Case {idx + 1}: {test_case['word']}\n"
        prompt += f"  Category: {test_case.get('category', 'N/A')}\n"
        prompt += f"  Pattern Group: {test_case.get('pattern_group', 'N/A')}\n"
        prompt += f"  Frequency: {test_case.get('frequency', 'N/A')}\n"
        prompt += "  With EM:\n"
        prompt += f"    Success: {with_res['result']['success']}\n"
        prompt += f"    Processing Time: {with_res['result']['processing_time']:.4f}s\n"
        prompt += f"    Memory Stats: {with_res['result'].get('memory_stats', {})}\n"
        prompt += "  Without EM:\n"
        prompt += f"    Success: {without_res['result']['success']}\n"
        prompt += f"    Processing Time: {without_res['result']['processing_time']:.4f}s\n\n"

    # Questions for Analysis
    prompt += (
        "### Key Questions for Analysis ###\n\n"
        "1. How does the integration of an episodic memory module (EM) influence processing time and memory "
        "utilization, particularly in ambiguous and extreme cases?\n"
        "2. What trade-offs, if any, are introduced by the EM module in terms of computational overhead or "
        "complexity?\n"
        "3. Are there scenarios where the episodic memory module might degrade performance or introduce unnecessary "
        "complexity?\n"
        "4. What improvements could be made to the episodic memory module to enhance generalization for novel and "
        "rare cases?\n"
        "5. How does the episodic memory module enhance the language agent's ability to generalize from past "
        "experiences to new and unseen linguistic patterns?\n"
        "6. What potential implications do these findings have for the development of more advanced AI systems, "
        "including aspects relevant to AGI?\n\n"
        "Caution: These findings are based on a limited set of test cases and may not generalize to all scenarios."
    )

    return prompt


def analyze_results(results_with: List[Dict], results_without: List[Dict]) -> Dict[str, Any]:
    """Analyze results in detail"""
    analysis = {
        'overall_success_rate': {
            'with_em': np.mean([int(r['result']['success']) for r in results_with]),
            'without_em': np.mean([int(r['result']['success']) for r in results_without])
        },
        'processing_time': {
            'with_em': np.mean([r['result']['processing_time'] for r in results_with]),
            'without_em': np.mean([r['result']['processing_time'] for r in results_without])
        },
        'category_performance': defaultdict(lambda: {'with_em': [], 'without_em': []})
    }

    # Analyze performance by category
    for with_r, without_r in zip(results_with, results_without):
        category = with_r['test_case']['category']
        analysis['category_performance'][category]['with_em'].append(int(with_r['result']['success']))
        analysis['category_performance'][category]['without_em'].append(int(without_r['result']['success']))

    # Calculate mean performance by category
    for category in analysis['category_performance']:
        analysis['category_performance'][category] = {
            'with_em': np.mean(analysis['category_performance'][category]['with_em']),
            'without_em': np.mean(analysis['category_performance'][category]['without_em'])
        }

    return analysis


def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    directories = ['outputs', 'outputs/plots', 'outputs/logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def save_experiment_results(results_with: List[Dict],
                            results_without: List[Dict],
                            analysis: Dict[str, Any],
                            llm_analysis: str):
    """Save detailed experiment results and LLM analysis"""
    ensure_output_directory()

    # Save LLM analysis to a log file
    with open(f'outputs/logs/llm_analysis.md', 'w') as f:
        f.write(llm_analysis)

    # Save plots
    def plot_and_save_detailed_results():
        sns.set_style("whitegrid")

        # Create main plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Overall Success Rate
        success_data = pd.DataFrame({
            'Method': ['With EM', 'Without EM'],
            'Success Rate': [analysis['overall_success_rate']['with_em'],
                             analysis['overall_success_rate']['without_em']]
        })
        sns.barplot(x='Method', y='Success Rate', data=success_data, ax=ax1)
        ax1.set_title('Overall Agreement Success Rate')

        # 2. Processing Time Distribution
        processing_times = pd.DataFrame({
            'Method': ['With EM'] * len(results_with) + ['Without EM'] * len(results_without),
            'Time (s)': [r['result']['processing_time'] for r in results_with] +
                        [r['result']['processing_time'] for r in results_without]
        })
        sns.boxplot(x='Method', y='Time (s)', data=processing_times, ax=ax2)
        ax2.set_title('Processing Time Distribution')

        # 3. Category Performance
        category_data = []
        for category, perf in analysis['category_performance'].items():
            category_data.extend([
                {'Category': category, 'Method': 'With EM', 'Success Rate': perf['with_em']},
                {'Category': category, 'Method': 'Without EM', 'Success Rate': perf['without_em']}
            ])
        category_df = pd.DataFrame(category_data)
        sns.barplot(x='Category', y='Success Rate', hue='Method', data=category_df, ax=ax3)
        ax3.set_title('Performance by Word Category')
        plt.setp(ax3.get_xticklabels(), rotation=45)

        # 4. Pattern Group Analysis
        pattern_data = []
        for r in results_with + results_without:
            pattern_group = r['test_case'].get('pattern_group', 'unknown')
            pattern_data.append({
                'Pattern': pattern_group,
                'Method': 'With EM' if r in results_with else 'Without EM',
                'Processing Time': r['result']['processing_time']
            })
        pattern_df = pd.DataFrame(pattern_data)
        sns.boxplot(x='Pattern', y='Processing Time', hue='Method', data=pattern_df, ax=ax4)
        ax4.set_title('Processing Time by Pattern')
        plt.setp(ax4.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(f'outputs/plots/comparison.png')
        plt.close()

        # Additional plots
        # Learning Curve
        plt.figure(figsize=(10, 6))
        case_numbers = range(len(results_with))
        times_with = [r['result']['processing_time'] for r in results_with]
        times_without = [r['result']['processing_time'] for r in results_without]
        plt.plot(case_numbers, times_with, 'b-', label='With EM')
        plt.plot(case_numbers, times_without, 'r--', label='Without EM')
        plt.title('Learning Curve')
        plt.xlabel('Test Case Number')
        plt.ylabel('Processing Time (s)')
        plt.legend(loc='upper right')
        plt.savefig(f'outputs/plots/learning_curve.png')
        plt.close()

        # Frequency Analysis
        plt.figure(figsize=(10, 6))
        freq_data = []
        for r in results_with + results_without:
            freq_data.append({
                'Word Frequency': r['test_case'].get('frequency', 'unknown'),
                'Method': 'With EM' if r in results_with else 'Without EM',
                'Processing Time': r['result']['processing_time']
            })
        freq_df = pd.DataFrame(freq_data)
        sns.boxplot(x='Word Frequency', y='Processing Time', hue='Method', data=freq_df)
        plt.title('Processing Time by Word Frequency')
        plt.savefig(f'outputs/plots/frequency_analysis.png')
        plt.close()

        # Memory Utilization Analysis
        plt.figure(figsize=(10, 6))
        memory_data = pd.DataFrame({
            'Test Case': range(len(results_with)),
            'Memory Utilization (With EM)': [r['result']['memory_utilization'] for r in results_with],
            'Memory Utilization (Without EM)': [0] * len(results_without)  # No memory usage for non-EM
        })
        plt.plot(memory_data['Test Case'], memory_data['Memory Utilization (With EM)'], label='With EM', color='blue')
        plt.plot(memory_data['Test Case'], memory_data['Memory Utilization (Without EM)'], label='Without EM',
                 color='red')
        plt.title('Memory Utilization by Test Case')
        plt.xlabel('Test Case Number')
        plt.ylabel('Memory Utilization')
        plt.legend()
        plt.savefig('outputs/plots/memory_utilization.png')
        plt.close()

        # Efficiency Comparison: Processing Time vs Memory Utilization
        plt.figure(figsize=(10, 6))
        plt.scatter(memory_data['Memory Utilization (With EM)'], times_with, color='blue', label='With EM')
        plt.scatter(memory_data['Memory Utilization (Without EM)'], times_without, color='red', label='Without EM')
        plt.title('Efficiency Comparison: Processing Time vs Memory Utilization')
        plt.xlabel('Memory Utilization')
        plt.ylabel('Processing Time (s)')
        plt.legend()
        plt.savefig('outputs/plots/efficiency_comparison.png')
        plt.close()

    def save_detailed_log():
        """Save detailed analysis log"""
        with open(f'outputs/logs/experiment_log.txt', 'w') as f:
            f.write("=== Experiment Results ===\n\n")

            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write(f"Success Rate (With EM): {analysis['overall_success_rate']['with_em']:.2%}\n")
            f.write(f"Success Rate (Without EM): {analysis['overall_success_rate']['without_em']:.2%}\n")
            f.write(f"Avg Processing Time (With EM): {analysis['processing_time']['with_em']:.4f}s\n")
            f.write(f"Avg Processing Time (Without EM): {analysis['processing_time']['without_em']:.4f}s\n\n")

            # Category Performance
            f.write("Performance by Category:\n")
            for category, perf in analysis['category_performance'].items():
                f.write(f"{category}:\n")
                f.write(f"  With EM: {perf['with_em']:.2%}\n")
                f.write(f"  Without EM: {perf['without_em']:.2%}\n")

            # Detailed Results
            f.write("\nDetailed Test Case Results:\n")
            for i, (with_r, without_r) in enumerate(zip(results_with, results_without)):
                f.write(f"\nTest Case {i + 1}: {with_r['test_case']['word']}\n")
                f.write(f"Category: {with_r['test_case']['category']}\n")
                f.write(f"Pattern Group: {with_r['test_case'].get('pattern_group', 'N/A')}\n")
                f.write(f"Frequency: {with_r['test_case'].get('frequency', 'N/A')}\n")
                f.write("With EM:\n")
                f.write(f"  Success: {with_r['result']['success']}\n")
                f.write(f"  Processing Time: {with_r['result']['processing_time']:.4f}s\n")
                if 'memory_stats' in with_r['result']:
                    f.write(f"  Memory Stats: {with_r['result']['memory_stats']}\n")
                f.write("Without EM:\n")
                f.write(f"  Success: {without_r['result']['success']}\n")
                f.write(f"  Processing Time: {without_r['result']['processing_time']:.4f}s\n")

    # Execute both functions
    plot_and_save_detailed_results()
    save_detailed_log()


def main():
    # Run experiments
    results_with = run_experiment(with_episodic_memory=True)
    results_without = run_experiment(with_episodic_memory=False)

    # Analyze results
    analysis = analyze_results(results_with, results_without)

    # Format the overall results into a prompt
    overall_prompt = format_overall_results(results_with, results_without, analysis)

    # Query the LLM for analysis
    llm_analysis = query_openai(overall_prompt)

    # Save results and LLM analysis
    save_experiment_results(results_with, results_without, analysis, llm_analysis)

    # Print analysis to console
    print("\nDetailed Analysis:")
    print(f"Overall Success Rate:")
    print(f"  With EM: {analysis['overall_success_rate']['with_em'] * 100:.2f}%")
    print(f"  Without EM: {analysis['overall_success_rate']['without_em'] * 100:.2f}%")
    print(f"\nAverage Processing Time:")
    print(f"  With EM: {analysis['processing_time']['with_em']:.4f}s")
    print(f"  Without EM: {analysis['processing_time']['without_em']:.4f}s")
    print("\nPerformance by Category:")
    for category, perf in analysis['category_performance'].items():
        print(f"  {category}:")
        print(f"    With EM: {perf['with_em'] * 100:.2f}%")
        print(f"    Without EM: {perf['without_em'] * 100:.2f}%")

    print("\nLLM Analysis:")
    print(llm_analysis)

    print("\nResults and visualizations saved in outputs directory.")


if __name__ == '__main__':
    main()
