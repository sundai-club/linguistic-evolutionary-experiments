import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import random

class QwenLoRATestor:
    def __init__(self, base_model_name="Qwen/Qwen3-8B", finetuned_path="./qwen_lora_final"):
        self.base_model_name = base_model_name
        self.finetuned_path = finetuned_path
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        # Load finetuned model
        print("Loading finetuned model...")
        self.finetuned_model = PeftModel.from_pretrained(
            self.base_model,
            finetuned_path
        )
        
    def generate_response(self, model, prompt, max_length=100):
        """Generate response using the given model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode only the generated part (excluding input)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
    
    def test_on_sample_prompts(self):
        """Test both models on sample prompts from training data format"""
        test_prompts = [
            "User: Describe this binary data pattern in exactly 8 words or fewer: 1010101010101010\nAssistant:",
            "User: Describe this binary data pattern in exactly 8 words or fewer: 1111111100000000\nAssistant:",
            "User: Describe this binary data pattern in exactly 8 words or fewer: 1001100110011001\nAssistant:",
            "User: Based on this description: 'Alternating ones and zeros', reconstruct the original binary string of length 16. Return only the binary string (0s and 1s).\nAssistant:",
            "User: Based on this description: 'First half ones, second half zeros', reconstruct the original binary string of length 16. Return only the binary string (0s and 1s).\nAssistant:"
        ]
        
        print("Testing models on sample prompts...\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"Test {i}: {prompt.split('User: ')[1].split('\\nAssistant:')[0]}")
            print("-" * 50)
            
            # Generate with base model
            base_response = self.generate_response(self.base_model, prompt)
            print(f"Base model: {base_response}")
            
            # Generate with finetuned model
            finetuned_response = self.generate_response(self.finetuned_model, prompt)
            print(f"Finetuned model: {finetuned_response}")
            
            print("\n")
    
    def test_on_training_data(self, data_file="openai_results.json", num_samples=5):
        """Test both models on actual training data examples"""
        try:
            with open(data_file, 'r') as f:
                training_data = json.load(f)
            
            # Select random samples
            samples = random.sample(training_data, min(num_samples, len(training_data)))
            
            print(f"Testing on {len(samples)} samples from training data...\n")
            
            for i, sample in enumerate(samples, 1):
                user_prompt = sample['user_prompt']
                expected_response = sample['agent_response']
                score = sample['score']
                
                # Format prompt for generation
                prompt = f"User: {user_prompt}\nAssistant:"
                
                print(f"Sample {i} (Score: {score}):")
                print(f"Prompt: {user_prompt}")
                print(f"Expected: {expected_response}")
                print("-" * 50)
                
                # Generate with base model
                base_response = self.generate_response(self.base_model, prompt)
                print(f"Base model: {base_response}")
                
                # Generate with finetuned model
                finetuned_response = self.generate_response(self.finetuned_model, prompt)
                print(f"Finetuned model: {finetuned_response}")
                
                print("\n")
                
        except FileNotFoundError:
            print(f"Training data file {data_file} not found. Skipping training data test.")
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("Interactive testing mode. Type 'quit' to exit.")
        
        while True:
            user_input = input("\nEnter your prompt: ")
            if user_input.lower() == 'quit':
                break
                
            prompt = f"User: {user_input}\nAssistant:"
            
            print("\nGenerating responses...")
            
            # Generate with base model
            base_response = self.generate_response(self.base_model, prompt)
            print(f"Base model: {base_response}")
            
            # Generate with finetuned model
            finetuned_response = self.generate_response(self.finetuned_model, prompt)
            print(f"Finetuned model: {finetuned_response}")

def main():
    print("Initializing model tester...")
    tester = QwenLoRATestor()
    
    print("=" * 60)
    print("SAMPLE PROMPTS TEST")
    print("=" * 60)
    tester.test_on_sample_prompts()
    
    print("=" * 60)
    print("TRAINING DATA TEST")
    print("=" * 60)
    tester.test_on_training_data()
    
    print("=" * 60)
    print("INTERACTIVE TEST")
    print("=" * 60)
    tester.interactive_test()

if __name__ == "__main__":
    main()