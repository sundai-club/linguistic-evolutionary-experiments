from openai_client import OpenAIClient
from dataset_builder import basic_dataset
from make_permutations import make_permutations
import json

def run_three_phase_experiment():
    """Run the three-phase constrained communication experiment"""
    # Generate dataset
    combos = make_permutations(4)
    dataset = basic_dataset(combos, 5)  # Small test dataset
    
    print("Generated dataset:", dataset)
    
    # Initialize OpenAI client
    client = OpenAIClient()
    
    # Run three-phase process
    results = client.three_phase_process(dataset, max_words=6)
    
    # Display results
    print("\n=== Three-Phase Results ===")
    for i, result in enumerate(results, 1):
        print(f"\nItem {i}:")
        print(f"  Original:      {result['original']}")
        print(f"  Description:   {result['description']}")
        print(f"  Reconstruction: {result['reconstruction']}")
        print(f"  Match:         {result['match']}")
        print(f"  Judge Correct: {result['judge_correct']}")
    
    # Calculate and display accuracy
    matches = sum(1 for result in results if result['match'])
    judge_correct = sum(1 for result in results if result['judge_correct'])
    accuracy = matches / len(results) if results else 0
    judge_accuracy = judge_correct / len(results) if results else 0
    
    print(f"\nString Match Accuracy: {accuracy:.2%} ({matches}/{len(results)})")
    print(f"Judge Accuracy: {judge_accuracy:.2%} ({judge_correct}/{len(results)})")
    
    return results

if __name__ == "__main__":
    run_three_phase_experiment()