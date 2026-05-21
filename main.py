from kvpress import SnapKVPress
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def main():
    model_name = "EleutherAI/pythia-70m"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    press = SnapKVPress(compression_ratio=0.5)

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    start = time.time()
    model.generate(**inputs, max_new_tokens=40)
    torch.cuda.synchronize()
    t_baseline = time.time() - start

    torch.cuda.synchronize()
    start = time.time()
    with press(model):
        model.generate(**inputs, max_new_tokens=40)
    torch.cuda.synchronize()
    t_snapkv = time.time() - start

    print("\n======= 实验结果 =======")
    print(f"基线耗时：{t_baseline:.3f} s")
    print(f"SnapKV 耗时：{t_snapkv:.3f} s")
    print(f"加速比：{t_baseline / t_snapkv:.2f} x")
    print("========================\n")

if __name__ == "__main__":
    main()