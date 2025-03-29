import random
import time
import weakref
from typing import Dict, List, Tuple, Optional
import heapq

import bittensor as bt
import numpy as np
from common import owner


class ValidatorStats:
    """Tracks statistics for individual validators."""
    
    def __init__(self) -> None:
        self.success_count = 0
        self.failure_count = 0
        self.total_fidelity_scores = 0.0
        self.last_success_time = 0
        self.last_failure_time = 0
        self.response_times: List[float] = []
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate, default to 0 if no attempts."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def avg_fidelity_score(self) -> float:
        """Calculate average fidelity score, default to 0 if no successes."""
        return self.total_fidelity_scores / self.success_count if self.success_count > 0 else 0.0
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time, default to a high value if no data."""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 300.0
    
    def record_success(self, timestamp: int, fidelity_score: float, response_time: float) -> None:
        """Record a successful task submission."""
        self.success_count += 1
        self.total_fidelity_scores += fidelity_score
        self.last_success_time = timestamp
        self.response_times.append(response_time)
        # Keep only the last 20 response times to adapt to changing conditions
        if len(self.response_times) > 20:
            self.response_times.pop(0)
    
    def record_failure(self, timestamp: int) -> None:
        """Record a failed task submission."""
        self.failure_count += 1
        self.last_failure_time = timestamp


class ValidatorSelector:
    """Encapsulates validator selection with optimized strategies for maximum rewards."""

    def __init__(self, metagraph: bt.metagraph, min_stake: int) -> None:
        self._metagraph_ref = weakref.ref(metagraph)
        self._min_stake = min_stake
        self._cooldowns: Dict[int, int] = {}
        
        # Track statistics for each validator
        self._validator_stats: Dict[int, ValidatorStats] = {}
        
        # Track the 4-hour observation window for optimizing task distribution
        self._observation_window = 4 * 60 * 60  # 4 hours in seconds
        self._successful_submissions: List[Tuple[int, int]] = []  # (timestamp, validator_uid)
        
        # Parameters for exploration vs exploitation
        self._exploration_rate = 0.2  # 20% of the time, explore new validators
        
        # Track the time remaining in the current 4h window
        self._current_window_end = int(time.time()) + self._observation_window
        
        # Temporary measure for subnet owner preference
        self._ask_owner_in = 5  # turns
        self._owner_hotkey = owner.HOTKEY
        if self._owner_hotkey not in metagraph.hotkeys:
            self._owner_uid = None
        else:
            self._owner_uid = metagraph.hotkeys.index(self._owner_hotkey)
            # Initialize stats for the owner
            self._validator_stats[self._owner_uid] = ValidatorStats()

    def get_next_validator_to_query(self) -> Optional[int]:
        """Get the next validator to query, optimized for maximum rewards."""
        current_time = int(time.time())
        metagraph: bt.metagraph = self._metagraph_ref()
        
        # Check if we need to update the 4h window
        if current_time >= self._current_window_end:
            self._current_window_end = current_time + self._observation_window
            # Prune old submissions outside the window
            self._prune_old_submissions(current_time - self._observation_window)
        
        # Subnet owner query logic (kept for compatibility)
        if self._query_subnet_owner(current_time):
            bt.logging.debug("Querying task from the subnet owner")
            return self._owner_uid
        
        # Exploration: Randomly select a validator sometimes to discover new opportunities
        if random.random() < self._exploration_rate:
            return self._explore_new_validator(metagraph, current_time)
        
        # Exploitation: Select the best validator based on our scoring system
        return self._exploit_best_validator(metagraph, current_time)

    def _explore_new_validator(self, metagraph: bt.metagraph, current_time: int) -> Optional[int]:
        """Randomly explore validators to discover new opportunities."""
        available_validators = [
            i for i in range(metagraph.n) 
            if metagraph.axons[i].is_serving 
            and metagraph.S[i] >= self._min_stake
            and self._cooldowns.get(i, 0) < current_time
            and (i not in self._validator_stats or self._validator_stats[i].failure_count < 5)
        ]
        
        if not available_validators:
            bt.logging.info("No available validators to explore.")
            return None
            
        selected_uid = random.choice(available_validators)
        bt.logging.debug(f"Exploring new validator [{selected_uid}]. Stake: {metagraph.S[selected_uid]}")
        return selected_uid

    def _exploit_best_validator(self, metagraph: bt.metagraph, current_time: int) -> Optional[int]:
        """Select the best validator based on scoring metrics."""
        scored_validators = []
        
        for uid in range(metagraph.n):
            # Skip validators that are not available
            if not metagraph.axons[uid].is_serving or metagraph.S[uid] < self._min_stake or self._cooldowns.get(uid, 0) >= current_time:
                continue
                
            # Calculate score based on performance metrics
            score = self._calculate_validator_score(uid, current_time)
            if score > 0:
                scored_validators.append((-score, uid))  # Negative for max-heap behavior
        
        if not scored_validators:
            bt.logging.info("No validators available with positive scores.")
            return self._explore_new_validator(metagraph, current_time)
            
        # Use a heap to efficiently get the highest scoring validator
        heapq.heapify(scored_validators)
        _, best_uid = heapq.heappop(scored_validators)
        
        bt.logging.debug(f"Selected best validator [{best_uid}]. Stake: {metagraph.S[best_uid]}")
        return best_uid

    def _calculate_validator_score(self, uid: int, current_time: int) -> float:
        """Calculate a score for a validator based on multiple factors."""
        if uid not in self._validator_stats:
            self._validator_stats[uid] = ValidatorStats()
            return 1.0  # New validators get a base score to encourage exploration
            
        stats = self._validator_stats[uid]
        
        # If validator has consistently failed, give it a low score
        if stats.failure_count > 10 and stats.success_rate < 0.3:
            return 0.1
            
        # Calculate recency factor - prefer validators with recent successes
        recency_factor = 1.0
        if stats.last_success_time > 0:
            time_since_last_success = current_time - stats.last_success_time
            recency_factor = max(0.5, 1.0 - (time_since_last_success / (12 * 3600)))  # Decay over 12 hours
            
        # Base score components
        success_component = stats.success_rate * 1.5
        fidelity_component = stats.avg_fidelity_score * 2.0
        
        # Response time component (faster is better)
        response_time_norm = min(1.0, 300.0 / max(1.0, stats.avg_response_time))
        
        # Submissions in current window component
        # Count how many submissions we've made to this validator in the current window
        submissions_in_window = sum(1 for ts, v_uid in self._successful_submissions if v_uid == uid)
        
        # We want to balance submissions across validators to maximize throughput
        # Lower score as we approach too many submissions to one validator
        balance_factor = max(0.5, 1.0 - (submissions_in_window / 10.0))
        
        # Combine all factors
        score = (
            success_component * 0.30 + 
            fidelity_component * 0.30 + 
            response_time_norm * 0.20 + 
            recency_factor * 0.10 +
            balance_factor * 0.10
        )
        
        return score

    def record_success(self, validator_uid: int, fidelity_score: float, response_time: float) -> None:
        """Record a successful task submission for a validator."""
        current_time = int(time.time())
        
        if validator_uid not in self._validator_stats:
            self._validator_stats[validator_uid] = ValidatorStats()
            
        self._validator_stats[validator_uid].record_success(current_time, fidelity_score, response_time)
        self._successful_submissions.append((current_time, validator_uid))
        
        # Prune old submissions outside the 4h window
        self._prune_old_submissions(current_time - self._observation_window)

    def record_failure(self, validator_uid: int) -> None:
        """Record a failed task submission for a validator."""
        current_time = int(time.time())
        
        if validator_uid not in self._validator_stats:
            self._validator_stats[validator_uid] = ValidatorStats()
            
        self._validator_stats[validator_uid].record_failure(current_time)

    def _prune_old_submissions(self, threshold_time: int) -> None:
        """Remove submissions that are older than the threshold time."""
        self._successful_submissions = [(ts, uid) for ts, uid in self._successful_submissions if ts >= threshold_time]

    def set_cooldown(self, validator_uid: int, cooldown_until: int) -> None:
        """Set a cooldown for a validator."""
        self._cooldowns[validator_uid] = cooldown_until

    def _query_subnet_owner(self, current_time: int) -> bool:
        """Determine if the subnet owner should be queried."""
        if self._owner_uid is None or self._cooldowns.get(self._owner_uid, 0) > current_time:
            return False

        if self._ask_owner_in > 1:
            self._ask_owner_in -= 1
            return False

        self._ask_owner_in = 5
        return True

    def get_performance_stats(self) -> Dict[str, float]:
        """Get overall performance statistics for monitoring."""
        total_success = sum(stats.success_count for stats in self._validator_stats.values())
        total_failure = sum(stats.failure_count for stats in self._validator_stats.values())
        avg_fidelity = (
            sum(stats.total_fidelity_scores for stats in self._validator_stats.values()) / total_success 
            if total_success > 0 else 0.0
        )
        
        return {
            "success_rate": total_success / (total_success + total_failure) if (total_success + total_failure) > 0 else 0.0,
            "avg_fidelity_score": avg_fidelity,
            "submissions_in_window": len(self._successful_submissions),
        }
