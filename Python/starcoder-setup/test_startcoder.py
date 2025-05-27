import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer
import gc
import psutil
from dotenv import load_dotenv
import os
import time

class GenerationCallback(BaseStreamer):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        self.last_token_time = self.start_time
        self.tokens_generated = 0

    def put(self, value):
        current_time = time.time()
        self.tokens_generated += 1

        if current_time - self.last_token_time >= 5:
            print(".", end="", flush=True)
            self.last_token_time = current_time
            print(f"Tokens generated: {self.tokens_generated} in {current_time - self.start_time:.1f} seconds")

def main():
    # Preventive memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

    load_dotenv()

    print("Stape 1: Checking Resources...")
    print(f"Total RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    if torch.cuda.is_available():
        print("GPU device: ", torch.cuda.get_device_name(0))
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Free GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    print("\nStep 2: Loading the model (CPU only)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "bigcode/starcoder",
            device_map="cpu",
            trust_remote_code=True,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        print("Model loaded successfully!")

        tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
        print("Tokenizer loaded successfully!")

        print("\nStep 4: Simple generation test...")
        prompt = "# Python function to calculate factorial"
        print(f"Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")
        print("Input tokens: ", inputs)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                # callbacks=[GenerationCallback()]
            )
            print("Output tokens: ", outputs)

        print("\Results :")
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_code)

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()