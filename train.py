import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import wandb
from typing import List, Dict, Any
import numpy as np
from accelerate import Accelerator

class RewardWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss function that weights the standard language modeling loss by reward scores
        """
        labels = inputs.get("labels")
        rewards = inputs.get("rewards")
        
        # Forward pass
        outputs = model(**{k: v for k, v in inputs.items() if k not in ["rewards"]})
        
        # Get logits and calculate loss manually
        logits = outputs.get("logits")
        
        if labels is not None:
            # Shift labels and logits for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross entropy loss for each sample
            batch_size = shift_logits.size(0)
            losses = []
            
            for i in range(batch_size):
                # Calculate loss for this sample
                sample_logits = shift_logits[i].view(-1, shift_logits.size(-1))
                sample_labels = shift_labels[i].view(-1)
                
                # Only calculate loss for non-padding tokens
                valid_mask = sample_labels != -100
                if valid_mask.sum() > 0:
                    sample_loss = F.cross_entropy(
                        sample_logits[valid_mask], 
                        sample_labels[valid_mask], 
                        reduction='mean'
                    )
                    
                    # Weight by reward (normalize rewards to be positive)
                    reward_weight = torch.sigmoid(torch.tensor(rewards[i], device=sample_loss.device))
                    weighted_loss = sample_loss * reward_weight
                    losses.append(weighted_loss)
                else:
                    losses.append(torch.tensor(0.0, device=shift_logits.device))
            
            # Average losses across batch
            loss = torch.stack(losses).mean()
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

class QwenLoRATrainer:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", data_file: str = "openai_results.json"):
        self.model_name = model_name
        self.data_file = data_file
        self.accelerator = Accelerator()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_and_preprocess_data(self) -> Dataset:
        """Load data from openai_results.json and preprocess for training"""
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        # Use all data (we'll weight by rewards in loss function)
        filtered_data = data
        
        processed_data = []
        for item in filtered_data:
            # Create training examples with system prompt, user prompt, and response
            system_prompt = item.get('system_prompt', '')
            user_prompt = item.get('user_prompt', '')
            response = item.get('agent_response', '')
            score = item.get('score', 0)
            
            # Format as conversation
            if system_prompt:
                text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: {response}"
            else:
                text = f"User: {user_prompt}\nAssistant: {response}"
            
            processed_data.append({
                'text': text,
                'score': score,
                'input_ids': None  # Will be tokenized later
            })
        
        return Dataset.from_list(processed_data)
    
    def tokenize_function(self, examples):
        """Tokenize the text data"""
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        # Keep the reward scores for loss weighting
        tokenized['rewards'] = examples['score']
        
        return tokenized
    
    def create_weighted_trainer(self, dataset: Dataset):
        """Create a trainer with sample weighting based on scores"""
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir="./qwen_lora_checkpoints",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to="wandb" if wandb.run else None,
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer with custom reward-weighted loss
        trainer = RewardWeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        return trainer
    
    def train(self):
        """Main training function"""
        print("Loading and preprocessing data...")
        dataset = self.load_and_preprocess_data()
        print(f"Loaded {len(dataset)} training examples")
        
        print("Creating trainer...")
        trainer = self.create_weighted_trainer(dataset)
        
        print("Starting training...")
        trainer.train()
        
        print("Saving final model...")
        trainer.save_model("./qwen_lora_final")
        self.tokenizer.save_pretrained("./qwen_lora_final")
        
        print("Training completed!")

def main():
    # Initialize wandb for experiment tracking
    wandb.init(
        project="qwen-lora-finetuning",
        config={
            "model": "Qwen/Qwen3-8B",
            "task": "reward-based-finetuning",
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 5e-5,
            "batch_size": 2,
            "epochs": 3
        }
    )
    
    # Create trainer and start training
    trainer = QwenLoRATrainer()
    trainer.train()
    
    wandb.finish()

if __name__ == "__main__":
    main()

