# Merge a LoRA adapter into its frozen base -> standalone model dir (loadable by every eval harness).
#   python merge_lora.py <base_dir> <adapter_dir> <out_dir>
import os, shutil, sys
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

base_dir, adapter_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
model = AutoModelForImageTextToText.from_pretrained(base_dir, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
for f in ["chat_template.jinja", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
          "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json"]:
    if os.path.exists(os.path.join(base_dir, f)):
        shutil.copy(os.path.join(base_dir, f), out)
open(os.path.join(out, "MERGE_INFO.txt"), "w").write(f"base: {base_dir}\nadapter: {adapter_dir}\n")
print("merged ->", out)
