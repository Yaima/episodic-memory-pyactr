import math
import time
import traceback

import pyactr as actr
import numpy as np
from typing import Dict, Any, Optional
import pyactr.utilities

from modules.episodic_memory import EpisodicMemory, Episode


class AgentModel:
    def __init__(self, subsymbolic: bool = True, latency: float = 0.05, use_episodic_memory: bool = True):
        """Initialize model with proper ACT-R parameters"""
        model_params = {
            "subsymbolic": subsymbolic,
            # Enables subsymbolic computations for retrieval time and activation calculations
            "retrieval_threshold": 0.35,  # Minimum activation value required for a chunk to be retrieved
            "latency_factor": latency,  # Controls the latency (time taken for retrieval) based on activation levels
            "decay": 0.7,  # Governs how quickly activation decays over time.
            "instantaneous_noise": 0.5,
            # Adds noise to retrieval activation to simulate variability in human cognition.
            "baselevel_learning": True,  # Enables the calculation of base-level activation using prior retrievals
            "strict_harvesting": False,  # Controls whether retrieval considers strict adherence to retrieval criteria
            "motor_prepared": True,  # Prepares the motor module, reducing latency for motor actions
            "emma_noise": False,  # Introduces variability in eye movement and manual action modules
            "optimized_learning": True,  # Simplifies and speeds up base-level learning calculations
            "utility_learning": True,
            # Enables reinforcement learning for production rules based on success and utility
            "partial_matching": False,  # Enables approximate matching when perfect chunk matches are unavailable
            "rule_firing": latency  # Controls the time taken for a rule to fire
        }

        self.model = actr.ACTRModel(**model_params)
        self.latency = latency
        self.use_episodic_memory = use_episodic_memory
        self.setup_chunk_types()

        if self.use_episodic_memory:
            self.episodic_memory = EpisodicMemory(self.model)
        else:
            self.episodic_memory = None

        self.define_productions()
        self.setup_buffers()

        # Override pyactr.utilities.baselevel_learning
        pyactr.utilities.baselevel_learning = self.custom_baselevel_learning

    @staticmethod
    def custom_baselevel_learning(current_time, times, bll, decay, activation=None, optimized_learning=False):
        """
        Custom implementation of base-level learning to handle edge cases.

        Parameters:
        - current_time: float, current simulation time.
        - times: list, list of prior retrieval times.
        - bll: bool, flag to enable base-level learning.
        - decay: float, decay rate for older activations.
        - activation: float or None, optional prior activation to combine.
        - optimized_learning: bool, if True, use a simplified optimized learning formula.

        Returns:
        - float: Base-level activation value.
        """
        if not bll:
            return activation if activation is not None else 0.0

        if len(times) == 0:
            return activation if activation is not None else 0.0

        times = np.array(times)
        time_differences = np.clip(current_time - times, a_min=1e-6, a_max=None)

        try:
            if not optimized_learning:
                B = math.log(np.sum(time_differences ** -decay))
            else:
                B = math.log(len(times) / (1 - decay)) - decay * math.log(np.max(time_differences))
        except (ValueError, RuntimeWarning) as e:
            print(f"[ERROR] Base-level learning failed: {e}. Adjusting times.")
            valid_time_differences = time_differences[time_differences > 1e-6]
            if len(valid_time_differences) > 0:
                B = math.log(np.sum(valid_time_differences ** -decay))
            else:
                B = 0.0

        if activation is not None:
            try:
                B = math.log(math.exp(B) + math.exp(activation))
            except OverflowError as e:
                print(f"[ERROR] Activation combination failed: {e}. Using maximum value.")
                B = max(B, activation)

        return B

    def setup_chunk_types(self):
        """Set up all chunk types needed by the model"""
        # Define chunk types for words and goals
        self.word_type = self.model.chunktype("word",
                                              "animacy category meaning number person phonology syncat context")

        self.goal_type = self.model.chunktype("goal_lexeme",
                                              "task category number person tense meaning")

        # Define chunk types for agreement results
        self.agreement_result_type = self.model.chunktype("agreement_result",
                                                          "number person success")

        # Define chunk types for episodic memory
        if self.use_episodic_memory:
            self.episode_type = self.model.chunktype(
                "episode",
                "context action reward number person meaning activation error_rate generalization_score "
                "retrieval_count success"
            )

            self.episodic_result_type = self.model.chunktype("episodic_result",
                                                             "number person meaning success")

    def define_productions(self):
        """Define production rules with proper variable binding"""
        base_prods = {}

        if self.use_episodic_memory:
            base_prods.update({
                "start": {
                    "rule": """
                        =g>
                            isa goal_lexeme
                            task "agree"
                        ==>
                        =g>
                            isa goal_lexeme
                            task "check_memory"
                    """,
                    "utility": 1
                },
                "check_episodic": {
                    "rule": """
                        =g>
                            isa goal_lexeme
                            task "check_memory"
                            meaning =m
                        ?retrieval>
                            buffer empty
                        ==>
                        +retrieval>
                            isa episode
                            meaning =m
                    """,
                    "utility": 1
                },
                "use_episodic": {
                    "rule": """
                        =g>
                            isa goal_lexeme
                            task "check_memory"
                        =retrieval>
                            isa episode
                            number =num
                            person =per
                            reward =r
                        ==>
                        =g>
                            isa goal_lexeme
                            task "finished"
                            number =num
                            person =per
                    """,
                    "utility": 1
                },
                "episodic_failed": {
                    "rule": """
                        =g>
                            isa goal_lexeme
                            task "check_memory"
                        ?retrieval>
                            state error
                        ==>
                        =g>
                            isa goal_lexeme
                            task "retrieve_subject"
                    """,
                    "utility": 1
                }
            })
        else:
            # Start directly with retrieval if episodic memory is not used
            base_prods.update({
                "start": {
                    "rule": """
                        =g>
                            isa goal_lexeme
                            task "agree"
                            meaning =m
                        ==>
                        =g>
                            isa goal_lexeme
                            task "retrieve_subject"
                            meaning =m
                    """,
                    "utility": 1
                }
            })

        # Common productions for both cases
        base_prods.update({
            "retrieve_subject": {
                "rule": """
                    =g>
                        isa goal_lexeme
                        task "retrieve_subject"
                        meaning =m
                    ==>
                    +retrieval>
                        isa word
                        meaning =m
                """,
                "utility": 1
            },
            "process_subject": {
                "rule": """
                    =g>
                        isa goal_lexeme
                        task "retrieve_subject"
                        meaning =m
                    =retrieval>
                        isa word
                        meaning =m
                        number =num
                        person =pers
                    ==>
                    =g>
                        isa goal_lexeme
                        task "finished"
                        number =num
                        person =pers
                        meaning =m
                """,
                "utility": 1
            },
            # Add retrieval failure handling
            "retrieval_failed": {
                "rule": """
                    =g>
                        isa goal_lexeme
                        task "retrieve_subject"
                    ?retrieval>
                        state error
                    ==>
                    =g>
                        isa goal_lexeme
                        task "use_episodic_memory"
                """,
                "utility": 1
            },
            "retrieve_from_episodic": {
                "rule": """
                    =g>
                        isa goal_lexeme
                        task "use_episodic_memory"
                        meaning =m
                    ==>
                    +retrieval>
                        isa episode
                        meaning =m
                """,
                "utility": 1
            },
            "use_episodic_memory": {
                "rule": """
                    =g>
                        isa goal_lexeme
                        task "use_episodic_memory"
                    =retrieval>
                        isa episode
                        number =num
                        person =per
                        reward =r
                    ==>
                    =g>
                        isa goal_lexeme
                        task "finished"
                        number =num
                        person =per
                """,
                "utility": 1
            },
            "episodic_memory_failed": {
                "rule": """
                    =g>
                        isa goal_lexeme
                        task "use_episodic_memory"
                    ?retrieval>
                        state error
                    ==>
                    =g>
                        isa goal_lexeme
                        task "failed"
                """,
                "utility": 1
            }
        })

        # Add all productions to model with error handling
        for name, prod in base_prods.items():
            try:
                self.model.productionstring(name=name, string=prod["rule"])
            except Exception as e:
                print(f"Error adding production '{name}': {e}")

    def setup_buffers(self):
        """Set up model buffers"""
        # Create declarative memory if not exists
        if not hasattr(self.model, 'decmem'):
            self.model.decmem = actr.DeclarativeMemory(self.model)

        # Initialize the retrieval buffer
        self.model.set_retrieval('retrieval')  # Correctly set retrieval buffer

        # Initialize goal buffers if not already set
        if not hasattr(self.model, 'goals'):
            self.model.goals = {}

    def setup_initial_chunks(self, test_word: Dict[str, Any]):
        # Clear buffers
        self.model.goal.clear()
        self.model.retrieval.clear()

        # Get meaning without extra spaces or quotes
        meaning = test_word.get('meaning', f'[[{test_word["word"]}]]').strip().strip('"')
        context = test_word.get('context', 'general')

        # Add word to declarative memory
        word_chunk = self.model.chunkstring(string=f"""
            isa word
            category "noun"
            meaning "{meaning}"
            context "{context}"
            number "{test_word['number']}"
            person "{test_word['person']}"
            phonology "{test_word.get('phonology', '/' + test_word['word'] + '/')}"
            syncat "subject"
            animacy "{test_word.get('animacy', 'inanimate')}"
        """)

        self.model.decmem.add(word_chunk)

        # Set initial goal with number and person
        goal_chunk = self.model.chunkstring(string=f"""
                isa goal_lexeme
                task "agree"
                category "verb"
                number "{test_word['number']}"
                person "{test_word['person']}"
                tense "{test_word.get('tense', 'present')}"
                meaning "{meaning}"
            """)
        self.model.goal.add(goal_chunk)

        # Print initial state
        print("\nInitial state:")
        print("Goal buffer:", self.model.goal)
        print("Retrieval buffer:", self.model.retrieval)
        print("Declarative memory:", self.model.decmem)

    def run_agreement_test(self, test_word: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run test case, simulate agent behavior, and analyze results.
        """
        start_time = time.time()

        # Setup declarative memory and goal chunks
        self.setup_initial_chunks(test_word)
        print(f"\nTesting agreement with {test_word['word']}:")
        print("Features:", {k: v for k, v in test_word.items() if k != 'word'})

        # Attempt episodic memory retrieval
        retrieved_episodes = []
        if self.episodic_memory:
            # Generate context vector based on test_word features
            context_vector = np.array([
                hash(test_word.get('context', '')),
                hash(test_word.get('pattern_group', '')),
                hash(test_word.get('frequency', '')),
            ])
            print(f"Generated context vector: {context_vector}")

            # Retrieve similar episodes
            retrieved_episodes = self.episodic_memory.retrieve_episodes(
                context_vector=context_vector, k=3, similarity_threshold=0.7
            )
            print(f"Retrieved {len(retrieved_episodes)} episodes.")

            if retrieved_episodes:
                # Filter retrieved episodes by context relevance
                relevant_episode = next(
                    (ep for ep in retrieved_episodes if ep.context.get('context') == test_word.get('context')),
                    None
                )

                if relevant_episode:
                    # Update test_word based on the most relevant episode
                    test_word['pattern_group'] = relevant_episode.context.get('pattern_group',
                                                                              test_word.get('pattern_group'))
                    test_word['frequency'] = relevant_episode.context.get('frequency', test_word.get('frequency'))
                    test_word['animacy'] = relevant_episode.context.get('animacy', test_word.get('animacy'))
                    test_word['context'] = relevant_episode.context.get('context', test_word.get('context',
                                                                                                 None))  # Safely get context
                    # Log changes for debugging
                    print(f"Updated test_word based on retrieved episode: {test_word}")
                else:
                    print("No relevant episode found; using default test_word features.")

        # Initialize state
        print("\nInitial state:")
        print("Goal buffer:", self.model.goal)
        print("Retrieval buffer:", self.model.retrieval)
        print("Declarative memory:", self.model.decmem)

        # Run simulation
        try:
            sim = self.model.simulation(initial_time=0.1, trace=True, gui=False)
            sim.run()
        except Exception as e:
            print(f"Simulation error: {e}")
            traceback.print_exc()

        # Analyze results
        results = self.analyze_results(test_word)
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time

        # Store episode if simulation was successful
        if results['success']:
            self.store_episode(test_word, results)
            if self.episodic_memory:
                results['memory_stats'] = self.episodic_memory.get_statistics()
            results['used_episodic'] = bool(retrieved_episodes)
        else:
            results['used_episodic'] = False  # No successful retrieval was used

        return results

    def analyze_results(self, test_word: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze final model state"""
        # Initialize results
        results = {
            'success': False,
            'verb_number': None,
            'verb_person': None,
            'used_episodic': self.use_episodic_memory
        }

        # Check the goal buffer for final values
        if self.model.goal:
            goal_chunk = next(iter(self.model.goal), None)
            if goal_chunk:
                number = getattr(goal_chunk, 'number', None)
                person = getattr(goal_chunk, 'person', None)

                if number is not None and person is not None:
                    results.update({
                        'verb_number': str(number).strip('"'),  # Remove quotes if present
                        'verb_person': str(person).strip('"')
                    })

                    # Check if agreement was successful
                    results['success'] = (
                            results['verb_number'] == test_word['number'] and
                            results['verb_person'] == test_word['person']
                    )

        results['productions_fired'] = [prod for prod in self.model.productions]

        return results

    def store_episode(self, test_word: Dict[str, Any], results: Dict[str, Any]):
        """Store agreement episode in episodic memory"""
        if not self.episodic_memory:
            return

        # Create rich context
        context = {
            'word': test_word['word'],
            'number': test_word['number'],
            'person': test_word['person'],
            'pattern_group': test_word.get('pattern_group', 'unknown'),
            'frequency': test_word.get('frequency', 'unknown'),
            'category': test_word.get('category', 'unknown'),
            'features': self.extract_context(test_word)
        }

        # Create episode
        episode = Episode(
            state=self.extract_context(test_word),
            action="agreement_test",
            reward=1.0 if results['success'] else 0.0,
            next_state=None,
            context=context,
            meaning=test_word['meaning']
        )

        # Store episode
        self.episodic_memory.store_episode(episode)

    def extract_context(self, test_word: Dict[str, Any]) -> np.ndarray:
        """Extract rich contextual features"""
        features = []

        # Phonological features
        phon = test_word.get('phonology', '')
        phon_features = [
            hash(phon) % 1000 / 1000,  # Hash the phonology string to get a unique feature
            len(phon) / 10  # Normalized length
        ]
        features.extend(phon_features)

        # Semantic features
        meaning = test_word.get('meaning', '')
        meaning_features = [
            hash(meaning) % 1000 / 1000  # Hash the meaning for uniqueness
        ]
        features.extend(meaning_features)

        # Category features
        category = test_word.get('category', 'unknown')
        category_features = [
            1 if category == 'regular' else 0,
            1 if category == 'irregular' else 0,
            1 if category == 'novel' else 0,
            1 if category == 'interference' else 0,
            1 if category == 'ambiguous' else 0
        ]
        features.extend(category_features)

        # Pattern group features
        pattern = test_word.get('pattern_group', 'unknown')
        pattern_features = [
            1 if pattern == 'simple_plural' else 0,
            1 if pattern == 'sibilant' else 0,
            1 if pattern == 'latin' else 0,
            1 if pattern == 'suppletive' else 0,
            1 if pattern == 'homonym' else 0
        ]
        features.extend(pattern_features)

        # Frequency features
        frequency = test_word.get('frequency', 'unknown')
        freq_features = [
            1 if frequency == 'high' else 0,
            1 if frequency == 'medium' else 0,
            1 if frequency == 'low' else 0,
            1 if frequency == 'none' else 0
        ]
        features.extend(freq_features)

        return np.array(features)
