

# rwsft_trainer.py
import torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer

BASE_MODEL = "Qwen/Qwen3-8b"

class RWTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # pop() so it is not fed to the model
        scores = inputs.pop("scores")          # (batch,)
        outputs = model(**inputs)
        
        # standard token-level NLL
        logits = outputs.logits[:, :-1].contiguous()
        labels = inputs["labels"][:, 1:].contiguous()
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        ).view(labels.size(0), -1).mean(-1)      # (batch,)

        # score-weighted mean
        loss = (scores * nll).mean()
        return (loss, outputs) if return_outputs else loss

def make_dataset(path):
    ds = load_dataset("json", data_files=path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(ex):
        toks = tokenizer(
            ex["prompt"] + ex["response"],
            truncation=True,
            max_length=1024,
        )
        toks["labels"] = toks["input_ids"].copy()
        toks["scores"] = ex["score"]          # keep scalar in batch
        return toks

    return ds.map(tokenize, remove_columns=ds.column_names)

if __name__ == "__main__":
    ds = make_dataset("openai_results.json")

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    trainer = RWTrainer(
        model=AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",     quantization_config=quantization_config

        ),
        train_dataset=ds,
        args=dict(
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            output_dir="rwsft_ckpt",
            report_to="tensorboard",
        ),
    )
    trainer.train()
    trainer.push_to_hub("your-username/llama-3-rwsft")


