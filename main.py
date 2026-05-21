from kvpress import SnapKVPress
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import time

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    press = SnapKVPress(compression_ratio=0.5)

    print("加载 wikitext 测试文本...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=True)
    
    text = ""
    for item in dataset:
        if len(item["text"]) > 500:
            text = item["text"]
            break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    print("\n=== 基线测试 ===")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        baseline_ppl = torch.exp(outputs.loss).item()
    torch.cuda.synchronize()
    baseline_time = time.time() - start

    print("\n=== SnapKV 测试 ===")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad(), press(model):
        outputs = model(**inputs, labels=inputs["input_ids"])
        snapkv_ppl = torch.exp(outputs.loss).item()
    torch.cuda.synchronize()
    snapkv_time = time.time() - start

    print("\n" + "="*50)
    print("          作业实验结果")
    print("="*50)
    print(f"模型: {model_name}")
    print(f"数据集: wikitext")
    print(f"基线 PPL: {baseline_ppl:.2f}")
    print(f"SnapKV PPL: {snapkv_ppl:.2f}")
    print(f"基线耗时: {baseline_time:.2f}s")
    print(f"SnapKV 耗时: {snapkv_time:.2f}s")
    print(f"加速比: {baseline_time / snapkv_time:.2f}x")
    print("="*50)

if __name__ == "__main__":
    main()
