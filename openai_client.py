import openai
import os
import json
import random
import asyncio
import re
from dataset_builder import basic_dataset, dataset_2d
from make_permutations import make_permutations

def remove_think_tags(text):
    """Remove <think></think> tags from text"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def calculate_reward(is_correct, token_count):
    """
    Calculate reward based on correctness and token length
    - Base: 1 if correct, 0 if incorrect
    - Length adjustment: 
      - At 25 tokens: reward = 0
      - From 25 to 50 tokens: linearly decrease to -2
      - From 0 to 25 tokens: linearly increase to +2
    """
    # Length-based reward (independent of correctness)
    if token_count <= 25:
        # Linear from +2 at 0 tokens to 0 at 25 tokens
        length_reward = 2 * (25 - token_count) / 25
    else:
        # Linear from 0 at 25 tokens to -2 at 50 tokens
        length_reward = -2 * (token_count - 25) / 25
        # Cap at -2 for tokens > 50
        length_reward = max(length_reward, -2)
    
    # Base correctness reward
    correctness_reward = 1 if is_correct else 0
    
    # If token count is exactly 25, reward is 0 regardless of correctness
    if token_count == 25:
        return 0
    
    # Otherwise combine correctness and length
    return correctness_reward + length_reward

class OpenAIClient:
    def __init__(self, api_key=None):
        self.client = openai.AsyncOpenAI(
            base_url="http://192.168.0.119:8000/v1",
            api_key="api_key" or os.getenv('OPENAI_API_KEY')
        )
    
    async def phase_1_describe(self, data_item, max_words=10):
        """Phase 1: Describe the data in a low number of words"""
        prompt = f"Describe this binary data pattern in exactly {max_words} words or fewer: {data_item}"
        prompt += "\\nothink" 
        try:
            response = await self.client.chat.completions.create(
                model="Qwen/Qwen3-8b-AWQ",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            return response.choices[0].message.content.strip(), token_usage
            
        except Exception as e:
            return f"Error: {str(e)}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    async def phase_2_reconstruct(self, description, original_length, distractor_samples=None):
        """Phase 2: Reconstruct original data from description with distractor samples"""
        base_prompt = f"Based on this description: '{description}', reconstruct the original binary string of length {original_length}."
        
        if distractor_samples:
            prompt = f"{base_prompt}\n\nHere are set of binary strings, one of which is the answer that you must choose\n"
            for i, sample in enumerate(distractor_samples, 1):
                prompt += f"{i}. {sample}\n"
            prompt += "\nReturn only the binary string (0s and 1s) that matches the description."
        else:
            prompt = f"{base_prompt} Return only the binary string (0s and 1s)."
        prompt +="\\nothink"
        try:
            response = await self.client.chat.completions.create(
                model="Qwen/Qwen3-8b-AWQ",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10000
            )
            
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            return response.choices[0].message.content.strip(), token_usage
            
        except Exception as e:
            return f"Error: {str(e)}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    async def phase_3_judge(self, ground_truth, model_response):
        """Phase 3: Judge if the model selected the ground truth response"""
        prompt = f"""You are a judge evaluating if a model correctly identified the ground truth binary string.

Ground truth: {ground_truth}
Model response: {model_response}

Did the model select the ground truth? Return ONLY a JSON object with a single boolean field:
{{"correct": true}} or {{"correct": false}}

The model is correct if its response contains or exactly matches the ground truth binary string."""
        
        try:
            response = await self.client.chat.completions.create(
                model="Qwen/Qwen3-8b-AWQ",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                return result.get("correct", False), token_usage
            except json.JSONDecodeError:
                # Fallback: simple string matching
                return ground_truth in model_response.replace(" ", ""), token_usage
            
        except Exception as e:
            # Fallback: simple string matching
            return ground_truth in model_response.replace(" ", ""), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    async def process_single_item(self, data_item, data_batch, max_words=10):
        """Process a single data item through all three phases"""
        # Phase 1: Describe
        description, phase1_tokens = await self.phase_1_describe(data_item, max_words)
        
        # Generate up to 4 random distractor samples from the batch
        other_samples = [item for item in data_batch if item != data_item]
        num_distractors = min(4, len(other_samples))
        distractor_samples = random.sample(other_samples, num_distractors) if num_distractors > 0 else []
        
        # Insert the original example at a random index in the distractor examples
        if distractor_samples:
            random_index = random.randint(0, len(distractor_samples))
            distractor_samples.insert(random_index, data_item)
        else:
            distractor_samples = [data_item]
        
        # Phase 2: Reconstruct with distractors
        reconstruction, phase2_tokens = await self.phase_2_reconstruct(description, len(data_item), distractor_samples)
        
        # Phase 3: Judge correctness
        judge_result, phase3_tokens = await self.phase_3_judge(data_item, reconstruction)
        
        # Clean responses by removing think tags
        clean_description = remove_think_tags(description)
        clean_reconstruction = remove_think_tags(reconstruction)
        
        # Calculate rewards for phase 1 and phase 2
        phase1_reward = calculate_reward(True, phase1_tokens["completion_tokens"])  # Phase 1 doesn't have correctness check
        phase2_reward = calculate_reward(judge_result, phase2_tokens["completion_tokens"])
        
        # Create phase 1 data structure
        phase1_data = {
            "system_prompt": "",  # No system prompt used
            "user_prompt": f"Describe this binary data pattern in exactly {max_words} words or fewer: {data_item}",
            "agent_response": clean_description,
            "score": phase1_reward
        }
        
        # Create phase 2 data structure
        base_prompt = f"Based on this description: '{description}', reconstruct the original binary string of length {len(data_item)}."
        if distractor_samples:
            phase2_prompt = f"{base_prompt}\n\nHere are set of binary strings, one of which is the answer that you must choose\n"
            for i, sample in enumerate(distractor_samples, 1):
                phase2_prompt += f"{i}. {sample}\n"
            phase2_prompt += "\nReturn only the binary string (0s and 1s) that matches the description."
        else:
            phase2_prompt = f"{base_prompt} Return only the binary string (0s and 1s)."
            
        phase2_data = {
            "system_prompt": "",  # No system prompt used
            "user_prompt": phase2_prompt,
            "agent_response": clean_reconstruction,
            "score": phase2_reward
        }
        
        return {
            "phase1": phase1_data,
            "phase2": phase2_data,
            "reward": 1 if judge_result else 0  # Just the basic correctness reward
        }

    async def three_phase_process(self, data_batch, max_words=10):
        """Execute all three phases for a batch of data in parallel"""
        tasks = [self.process_single_item(data_item, data_batch, max_words) for data_item in data_batch]
        results = await asyncio.gather(*tasks)
        return results
    
    async def two_phase_process(self, data_batch, max_words=10):
        """Execute both phases for a batch of data in parallel (legacy method)"""
        return await self.three_phase_process(data_batch, max_words)
   
async def main():
    # Generate dataset
    combos = make_permutations(100)
    dataset = basic_dataset(combos, 100)
    
    # Initialize OpenAI client
    client = OpenAIClient()
    
    # Process using two-phase approach
    batch_size = 5
    all_results = []
    
    batch_tasks = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch_tasks.append(client.two_phase_process(batch, max_words=8))
    
    # Run all batches in parallel
    batch_results = await asyncio.gather(*batch_tasks)
    
    # Flatten results
    for i, results in enumerate(batch_results):
        all_results.extend(results)
        print(f"Processed two-phase batch {i + 1}")
    
    # Calculate accuracy
    correct_judgments = sum(1 for result in all_results if result['reward'] == 1)
    accuracy = correct_judgments / len(all_results) if all_results else 0
    
    # Prepare output data with phase 1 and phase 2 completions
    output_data = []
    for result in all_results:
        output_data.append(result['phase1'])
        output_data.append(result['phase2'])
    
    # Save results
    with open('openai_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Completed two-phase processing of {len(dataset)} items")
    print(f"Reconstruction accuracy: {accuracy:.2%} ({correct_judgments}/{len(all_results)})")
    print(f"Saved {len(output_data)} entries to openai_results.json")

if __name__ == "__main__":
    asyncio.run(main())
