import asyncio
import base64
import json
import random
import re
import time
import typing
import urllib.parse
from collections import deque

import aiohttp
import bittensor as bt
from aiohttp import ClientTimeout
from aiohttp.helpers import sentinel
from common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
from common.protocol import PullTask, SubmitResults

from miner import ValidatorSelector


NETWORK_DELAY_TIME_BUFFER = 60
FAILED_VALIDATOR_DELAY = 300
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 600  # Default timeout if submit_before not available
# Keep track of successful generations per endpoint for load balancing
GENERATION_SUCCESS_HISTORY = {}
# Track last N generations for quality analysis
RECENT_GENERATIONS = deque(maxlen=50)


async def worker_routine(
    endpoint: str, wallet: bt.wallet, metagraph: bt.metagraph, validator_selector: ValidatorSelector
) -> None:
    bt.logging.info(f"Worker ({endpoint}) started")
    generate_url = urllib.parse.urljoin(endpoint, "/generate/")
    
    # Initialize success history for this endpoint
    if endpoint not in GENERATION_SUCCESS_HISTORY:
        GENERATION_SUCCESS_HISTORY[endpoint] = {
            "success": 0,
            "failure": 0,
            "avg_quality": 0.0,
            "last_success_time": 0
        }

    while True:
        await _complete_one_task(generate_url, wallet, metagraph, validator_selector)


async def _complete_one_task(
    generate_url: str, wallet: bt.wallet, metagraph: bt.metagraph, validator_selector: ValidatorSelector
) -> None:
    validator_uid = validator_selector.get_next_validator_to_query()
    if validator_uid is None:
        await asyncio.sleep(10.0)
        return

    dendrite = bt.dendrite(wallet=wallet)

    # Setting cooldown to prevent selecting the same validator for concurrent task.
    validator_selector.set_cooldown(validator_uid, int(time.time()) + 60)

    # Record the start time for response time measurement
    start_time = time.time()
    
    pull = await _pull_task(dendrite, metagraph, validator_uid)
    if pull.dendrite.status_code != 200:
        bt.logging.warning(
            f"Failed to get task from [{metagraph.hotkeys[validator_uid]}]. Reason: {pull.dendrite.status_message}."
        )
        # Record failure
        validator_selector.record_failure(validator_uid)
        validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
        return

    if pull.task is None:
        if not hasattr(pull, 'cooldown_until') or pull.cooldown_until == 0:
            bt.logging.warning(f"Failed to get task from [{metagraph.hotkeys[validator_uid]}]. Reason: Unknown.")
            # Record failure
            validator_selector.record_failure(validator_uid)
            validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
        else:
            cooldown_left = max(0, int(pull.cooldown_until - time.time()))
            violations = getattr(pull, 'cooldown_violations', 0)
            bt.logging.debug(
                f"Miner is on cooldown for the next: {cooldown_left} sec. "
                f"Total cooldown violations: {violations}"
            )
            validator_selector.set_cooldown(validator_uid, pull.cooldown_until)
        return

    bt.logging.debug(f"Task received. Prompt: {pull.task.prompt}")

    # Updating cooldown. Validator won't give new tasks until this one is submitted.
    # Use a default timeout if submit_before is not available
    next_cooldown = int(time.time()) + 300  # Default 5 minute cooldown
    if hasattr(pull, 'submit_before') and pull.submit_before > 0:
        next_cooldown = pull.submit_before
    validator_selector.set_cooldown(validator_uid, next_cooldown)

    # Calculate maximum time for generation with buffer for network delays
    timeout = None
    if hasattr(pull, 'submit_before') and pull.submit_before > 0:
        timeout = pull.submit_before - time.time() - NETWORK_DELAY_TIME_BUFFER
    else:
        timeout = DEFAULT_TIMEOUT  # Use default timeout
    
    # Optimize prompt based on historical data
    optimized_prompt = _optimize_prompt(pull.task.prompt, validator_uid)
    
    # Generate with enhanced process
    results = await _enhanced_generate(generate_url, optimized_prompt, timeout, validator_uid)
    if results is None:
        # Record failure if generation failed
        validator_selector.record_failure(validator_uid)
        return

    # Apply post-processing to improve quality
    processed_results = _post_process_results(results, pull.task.prompt, validator_uid)
    
    # Check quality before submitting
    quality_score = _evaluate_generation_quality(processed_results, pull.task.prompt)
    if quality_score < 0.5:  # Minimum quality threshold
        bt.logging.warning(f"Generated result quality too low ({quality_score:.2f}). Regenerating...")
        # Try one more time with a more detailed prompt
        enhanced_prompt = _enhance_prompt_for_retry(pull.task.prompt)
        results = await _enhanced_generate(generate_url, enhanced_prompt, timeout, validator_uid)
        if results is None:
            validator_selector.record_failure(validator_uid)
            return
        processed_results = _post_process_results(results, pull.task.prompt, validator_uid)
    
    submit = await _submit_results(wallet, dendrite, metagraph, validator_uid, pull, processed_results)
    # Calculate total response time for this task
    response_time = time.time() - start_time
    
    if submit.feedback is None:
        bt.logging.warning(
            f"Failed to submit results to [{metagraph.hotkeys[validator_uid]}]. "
            f"Reason: {submit.dendrite.status_message}."
        )
        # Record failure
        validator_selector.record_failure(validator_uid)
        validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
        return

    _log_feedback(validator_uid, submit)
    
    # Record success with the feedback information
    if submit.feedback.task_fidelity_score > 0:
        validator_selector.record_success(
            validator_uid, 
            submit.feedback.task_fidelity_score,
            response_time
        )
        
        # Store this generation in our history for learning
        RECENT_GENERATIONS.append({
            "prompt": pull.task.prompt,
            "results": processed_results,
            "fidelity_score": submit.feedback.task_fidelity_score,
            "validator_uid": validator_uid,
            "timestamp": time.time()
        })
        
        # Update our endpoint success history
        endpoint_name = generate_url
        if endpoint_name in GENERATION_SUCCESS_HISTORY:
            history = GENERATION_SUCCESS_HISTORY[endpoint_name]
            history["success"] += 1
            history["avg_quality"] = (history["avg_quality"] * (history["success"] - 1) + 
                                      submit.feedback.task_fidelity_score) / history["success"]
            history["last_success_time"] = time.time()
    else:
        # Record as failure if validation failed
        validator_selector.record_failure(validator_uid)
        # Also record endpoint failure
        endpoint_name = generate_url
        if endpoint_name in GENERATION_SUCCESS_HISTORY:
            GENERATION_SUCCESS_HISTORY[endpoint_name]["failure"] += 1

    # Log performance stats periodically
    if random.random() < 0.05:  # ~5% chance to log stats
        stats = validator_selector.get_performance_stats()
        bt.logging.info(f"Performance stats: Success rate: {stats['success_rate']:.2f}, " 
                       f"Avg fidelity: {stats['avg_fidelity_score']:.2f}, "
                       f"Submissions in window: {stats['submissions_in_window']}")

    validator_selector.set_cooldown(validator_uid, submit.cooldown_until)


async def _pull_task(dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int) -> PullTask:
    synapse = PullTask()
    response = typing.cast(
        PullTask,
        await dendrite.call(
            target_axon=metagraph.axons[validator_uid], synapse=synapse, deserialize=False, timeout=12.0
        ),
    )
    return response


async def _submit_results(
    wallet: bt.wallet, dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int, pull: PullTask, results: str
) -> SubmitResults:
    submit_time = time.time_ns()
    prompt = pull.task.prompt if pull.task is not None else None
    message = (
        f"{MINER_LICENSE_CONSENT_DECLARATION}"
        f"{submit_time}{prompt}{metagraph.hotkeys[validator_uid]}{wallet.hotkey.ss58_address}"
    )
    signature = base64.b64encode(dendrite.keypair.sign(message)).decode(encoding="utf-8")
    synapse = SubmitResults(task=pull.task, results=results, submit_time=submit_time, signature=signature)
    response = typing.cast(
        SubmitResults,
        await dendrite.call(
            target_axon=metagraph.axons[validator_uid],
            synapse=synapse,
            deserialize=False,
            timeout=300.0,
        ),
    )
    return response


def _log_feedback(validator_uid: int, submit: SubmitResults) -> None:
    feedback = submit.feedback
    if feedback is None:
        return
    score = "failed" if feedback.validation_failed else feedback.task_fidelity_score
    bt.logging.debug(f"Feedback received from [{validator_uid}]. Prompt: {submit.task.prompt}. Score: {score}")
    bt.logging.debug(
        f"Average score: {feedback.average_fidelity_score}. "
        f"Accepted results (last 4h): {feedback.generations_within_the_window}. "
        f"Reward: {feedback.current_miner_reward}."
    )


async def _enhanced_generate(
    generate_url: str, prompt: str, timeout: float | None = None, validator_uid: int = None
) -> str | None:  # noqa: ASYNC109
    """Enhanced version of the generate function with retries and fallback strategy."""
    bt.logging.debug(f"Generating for prompt: {prompt} with timeout {timeout} seconds")
    
    # Prepare timeout with a buffer for pre/post processing
    buffer_seconds = 5  # Buffer time for processing
    if timeout is not None:
        adjusted_timeout = max(1.0, timeout - buffer_seconds)
    else:
        adjusted_timeout = None
    
    client_timeout = ClientTimeout(total=adjusted_timeout) if adjusted_timeout is not None else sentinel
    
    # Try multiple times with exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(generate_url, data={"prompt": prompt}) as response:
                    if response.status == 200:
                        results = await response.text()
                        bt.logging.debug(f"Generation completed on attempt {attempt}. Size: {len(results)}")
                        return results
                    else:
                        bt.logging.error(f"Generation failed with code: {response.status} on attempt {attempt}")
                        
                        # If we're out of retries, return None
                        if attempt == MAX_RETRIES:
                            return None
                            
                        # Add exponential backoff between retries
                        backoff_time = min(2.0 ** (attempt - 1), 10.0)  # Max 10 seconds
                        await asyncio.sleep(backoff_time)
        except (aiohttp.ClientConnectorError, TimeoutError, aiohttp.ClientError, Exception) as e:
            error_type = type(e).__name__
            bt.logging.error(f"{error_type} on attempt {attempt}: {str(e)} ({generate_url})")
            
            # If we're out of retries, return None
            if attempt == MAX_RETRIES:
                return None
                
            # Add exponential backoff between retries
            backoff_time = min(2.0 ** (attempt - 1), 10.0)  # Max 10 seconds
            await asyncio.sleep(backoff_time)
    
    return None


def _optimize_prompt(prompt: str, validator_uid: int) -> str:
    """Optimize the prompt based on historical data and validator preferences."""
    # Find successful generations for this validator
    successful_generations = [g for g in RECENT_GENERATIONS 
                            if g["validator_uid"] == validator_uid and g["fidelity_score"] > 0.7]
    
    if not successful_generations:
        # No history for this validator, return original prompt
        return prompt
    
    # Sort by fidelity score
    successful_generations.sort(key=lambda x: x["fidelity_score"], reverse=True)
    
    # Take inspiration from top performing prompts
    if len(successful_generations) >= 3:
        # If we have enough data, add instructions based on successful patterns
        high_quality_prompts = [g["prompt"] for g in successful_generations[:3]]
        
        # Simple template enhancement
        if any("detailed" in p.lower() for p in high_quality_prompts):
            prompt = f"{prompt}\nPlease provide a detailed response."
        
        if any("step by step" in p.lower() for p in high_quality_prompts):
            prompt = f"{prompt}\nConsider providing a step-by-step explanation."
    
    return prompt


def _post_process_results(results: str, prompt: str, validator_uid: int) -> str:
    """Apply post-processing to improve the quality of results."""
    # If results are very short, they might be incomplete
    if len(results) < 50:
        bt.logging.warning(f"Generation result suspiciously short ({len(results)} chars), may be incomplete")
        return results  # Return as is, might be a special case
    
    # Check if we need to clean up the results (remove artifacts, fix formatting)
    # These are common issues in generated text that can affect validator scoring
    
    # Remove excessive newlines
    results = re.sub(r'\n{3,}', '\n\n', results)
    
    # Check for JSON/code block completeness
    if results.count('{') != results.count('}'):
        bt.logging.warning("Detected unbalanced JSON brackets, attempting to fix")
        # Simple fix for common case of truncated JSON
        if results.count('{') > results.count('}'):
            missing_count = results.count('{') - results.count('}')
            results += '}' * missing_count
    
    # Check for code block completeness (common issue)
    if results.count('```') % 2 != 0:
        bt.logging.warning("Detected unclosed code blocks, attempting to fix")
        results += '\n```'
    
    return results


def _evaluate_generation_quality(results: str, prompt: str) -> float:
    """Evaluate the quality of generated results to determine if we should retry."""
    # Simple heuristics for quality evaluation
    
    # Check for minimal length (based on prompt complexity)
    expected_min_length = len(prompt) * 1.5  # Very simple heuristic
    if len(results) < expected_min_length and len(results) < 200:
        bt.logging.debug(f"Generation seems too short: {len(results)} chars")
        return 0.3
    
    # Check for repetition (a common issue in generation)
    words = results.split()
    if len(words) > 20:
        # Check for repetitive phrases (3+ word sequences)
        phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        phrase_count = {}
        for phrase in phrases:
            phrase_count[phrase] = phrase_count.get(phrase, 0) + 1
        
        # If any phrase appears more than 3 times, penalize
        if any(count > 3 for count in phrase_count.values()):
            bt.logging.debug("Detected excessive repetition in generation")
            return 0.4
    
    # Check for common error messages in the output
    error_indicators = ['error:', 'exception:', 'traceback:', 'could not generate', 'failed to']
    if any(indicator in results.lower() for indicator in error_indicators):
        bt.logging.debug("Detected possible error message in generation")
        return 0.2
    
    # If we pass all checks, assign a good baseline score
    return 0.8


def _enhance_prompt_for_retry(prompt: str) -> str:
    """Create an enhanced prompt for retry in case of low quality."""
    # Add specific instructions to improve generation
    enhanced_prompt = (
        f"{prompt}\n\n"
        "Please provide a comprehensive, detailed response. "
        "Ensure the answer is complete and addresses all aspects of the question. "
        "If there are multiple points to cover, please address each one thoroughly."
    )
    return enhanced_prompt
