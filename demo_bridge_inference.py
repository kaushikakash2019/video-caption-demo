import os
from glob import glob
import torch
from PIL import Image
from transformers import AutoProcessor, GitForCausalLM
from evaluate import load as load_metric

def main():
    # 1) Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 2) Load model & processor
    model_name = "microsoft/git-base"
    model     = GitForCausalLM.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    # 3) Gather demo frames
    demo_dir     = "data/bridgedata_v2"
    video_folder = os.path.join(demo_dir, "videos", "0001")
    frame_paths  = sorted(glob(os.path.join(video_folder, "*.png")))
    frames       = [Image.open(p).convert("RGB") for p in frame_paths]

    # 4) Preprocess with a text prompt
    inputs = processor(images=frames, text="Describe the video", return_tensors="pt").to(device)

    # 5) Generate caption
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50
    )
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Print result
    print("\nGenerated Caption:")
    print(preds[0])

    # 6) (Optional) Compute BLEU
    gt_path = os.path.join(demo_dir, "captions", "0001.txt")
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            gt = f.read().strip()
        bleu = load_metric("bleu")
        result = bleu.compute(predictions=[preds[0]], references=[[gt]])
        print(f"\nBLEU score: {result['bleu']:.4f}")

if __name__ == "__main__":
    main()
