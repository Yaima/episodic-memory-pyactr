import time
from collections import defaultdict
from typing import List, Dict, Any, Optional

import numpy as np


class Episode:
    def __init__(self,
                 state: np.ndarray,
                 action: str,
                 reward: float,
                 next_state: Optional[np.ndarray] = None,
                 context: Optional[Dict] = None,
                 meaning: Optional[str] = None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.context = context or {}
        self.meaning = meaning
        self.create_time = time.time()
        self.retrieval_count = 0
        self.last_access_time = self.create_time
        self.activation = 1.0
        self.error_history = []
        self.generalization_score = 0.0

    def get_context_vector(self) -> np.ndarray:
        """Return enriched context vector with temporal features"""
        base_vector = self.state

        # Add temporal features
        time_since_creation = time.time() - self.create_time
        time_since_last_access = time.time() - self.last_access_time

        temporal_features = np.array([
            1.0 / (1.0 + time_since_creation),  # Recency
            self.retrieval_count / (1.0 + time_since_creation),  # Frequency
            1.0 / (1.0 + time_since_last_access),  # Access recency
            self.activation  # Current activation
        ])

        # Add performance features if available
        if self.error_history:
            error_rate = sum(self.error_history) / len(self.error_history)
            performance_features = np.array([
                1.0 - error_rate,  # Success rate
                self.generalization_score
            ])
        else:
            performance_features = np.array([1.0, 0.0])

        # Combine all features
        return np.concatenate([base_vector, temporal_features, performance_features])

    def to_chunk_string(self) -> str:
        """Convert episode to enhanced ACT-R chunk string format"""
        return f"""
        isa episode
        context "{str(self.state.tolist())}"
        action "{str(self.action)}"
        reward "{str(self.reward)}"
        meaning "{str(self.meaning)}"
        number "{str(self.context.get('number', ''))}"
        person "{str(self.context.get('person', ''))}"
        retrieval_count "{str(self.retrieval_count)}"
        activation "{str(self.activation)}"
        generalization_score "{str(self.generalization_score)}"
        error_rate "{str(sum(self.error_history) / len(self.error_history) if self.error_history else 0.0)}"
        """

    def update_retrieval_stats(self, success: bool = True, similarity_score: float = 0.0):
        """Update episode statistics with disambiguation adjustments."""
        current_time = time.time()
        time_since_last = current_time - self.last_access_time

        # Update retrieval count and last access time
        self.retrieval_count += 1
        self.last_access_time = current_time

        # Update activation using ACT-R decay
        decay_rate = 0.5
        self.activation *= np.power(time_since_last + 1, -decay_rate)
        self.activation += 1.0  # Boost from current retrieval

        # Update error history for learning
        self.error_history.append(0 if success else 1)

        # Track disambiguation success
        if 'pattern_group' in self.context and self.context['pattern_group'] == 'homonym':
            self.generalization_score = max(self.generalization_score, similarity_score)


class EpisodicMemory:
    def __init__(self, model, max_episodes: int = 1000):
        self.episodes: List[Episode] = []
        self.model = model
        self.max_episodes = max_episodes
        self.creation_time = time.time()
        self.total_retrievals = 0
        self.successful_retrievals = 0
        self.memory_utilization = 0.0

        if self.model:
            self.model.chunktype(
                'episode',
                'context action reward number person meaning success activation retrieval_count error_rate'
            )

    def store_episode(self, episode: Episode):
        """Store episode with enhanced memory management"""
        # Adjust similarity threshold or skip similarity check
        # to ensure episodes are stored
        # Optionally, you can remove the similarity check:

        # Memory management if needed
        if len(self.episodes) >= self.max_episodes:
            self._cleanup_memory()

        self.episodes.append(episode)
        self.memory_utilization = len(self.episodes) / self.max_episodes

        # Store in ACT-R declarative memory
        if self.model:
            try:
                chunk = self.model.chunkstring(string=episode.to_chunk_string())
                if hasattr(self.model, 'dm'):
                    self.model.dm.add(chunk)
                else:
                    self.model.decmem.add(chunk)
            except Exception as e:
                print(f"Error storing episode chunk: {e}")

    def retrieve_episodes(self, context_vector: np.ndarray, k: int = 5, similarity_threshold: float = 0.7) -> List[
        Episode]:
        """Retrieve similar episodes with dynamic thresholds."""
        if not self.episodes:
            return []

        self.total_retrievals += 1
        similarities = []

        for episode in self.episodes:
            similarity = self._calculate_similarity(episode, context_vector)
            dynamic_threshold = 0.5 + 0.2 * (1 - self.memory_utilization)  # Adjust threshold dynamically
            if similarity >= max(dynamic_threshold, similarity_threshold):  # Use stricter of two thresholds
                similarities.append((similarity, episode))

        similarities.sort(reverse=True, key=lambda x: x[0])
        retrieved_episodes = [ep for _, ep in similarities[:k]]

        for episode in retrieved_episodes:
            episode.update_retrieval_stats(success=True)

        self.successful_retrievals += len(retrieved_episodes)
        return retrieved_episodes

    def _find_similar_episodes(self, target_episode: Episode) -> List[Episode]:
        """Find similar episodes"""
        similarities = []
        target_features = target_episode.get_context_vector()

        for episode in self.episodes:
            similarity = self._calculate_similarity(episode, target_features)
            if similarity >= 0.8:  # High similarity threshold
                similarities.append((similarity, episode))

        similarities.sort(reverse=True, key=lambda x: x[0])
        return [ep for _, ep in similarities]

    def _calculate_similarity(self, episode: Episode, context_vector: np.ndarray) -> float:
        """Calculate similarity between episode and context"""
        episode_vector = episode.get_context_vector()

        # Ensure vectors have same length
        min_length = min(len(episode_vector), len(context_vector))
        episode_vector = episode_vector[:min_length]
        context_vector = context_vector[:min_length]

        if len(episode_vector) == 0 or len(context_vector) == 0:
            return 0.0

        # Calculate cosine similarity
        dot_product = np.dot(episode_vector, context_vector)
        norm1 = np.linalg.norm(episode_vector)
        norm2 = np.linalg.norm(context_vector)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Weight by recency
        time_weight = 1.0 / (1.0 + (time.time() - episode.create_time) / 3600.0)

        # Return weighted similarity
        return similarity * (0.7 + 0.3 * time_weight)

    def _cleanup_memory(self):
        """Clean up memory based on utility"""
        if not self.episodes:
            return

        # Calculate utility scores
        utilities = []
        for ep in self.episodes:
            utility = (
                    ep.activation * 0.4 +  # Activation level
                    (ep.retrieval_count / 10.0) * 0.3 +  # Retrieval frequency
                    ep.reward * 0.3  # Success weighting
            )
            utilities.append((utility, ep))

        # Keep highest utility episodes
        utilities.sort(reverse=True)
        self.episodes = [ep for _, ep in utilities[:self.max_episodes - 1]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics with disambiguation metrics."""
        if not self.episodes:
            return {
                'total_episodes': 0,
                'memory_utilization': 0.0,
                'retrieval_accuracy': 0.0,
                'pattern_performance': {}
            }

        current_time = time.time()
        total_time = current_time - self.creation_time

        stats = {
            'total_episodes': len(self.episodes),
            'memory_utilization': self.memory_utilization,
            'retrieval_accuracy': self.successful_retrievals / max(1, self.total_retrievals),
            'avg_activation': np.mean([ep.activation for ep in self.episodes]),
            'avg_retrieval_count': np.mean([ep.retrieval_count for ep in self.episodes]),
            'memory_age': total_time,
            'retrieval_rate': self.total_retrievals / total_time if total_time > 0 else 0
        }

        # Add pattern-level success rates
        pattern_groups = defaultdict(list)
        for ep in self.episodes:
            if ep.context and 'pattern_group' in ep.context:
                pattern_groups[ep.context['pattern_group']].append(
                    sum(ep.error_history) / len(ep.error_history) if ep.error_history else 0.0
                )

        stats['pattern_performance'] = {
            pattern: np.mean(errors)
            for pattern, errors in pattern_groups.items()
        }

        return stats
