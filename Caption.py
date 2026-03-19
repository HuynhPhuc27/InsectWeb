# This is file for generating the caption of images, which then will be stored with .txt format. 
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import os

model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model & processor
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
prompt = "USER: <image>\nDescribe this insect in detail.\nASSISTANT:"


image_dir = "..."

caption_dir = "..."
os.makedirs(caption_dir, exist_ok=True)

image_files = sorted(
    [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))],
    reverse=True
)

start_from = "..."

start_processing = False

for img_file in image_files:
    if not start_processing:
        if img_file == start_from:
            start_processing = True
        else:
            continue
    try:
        img_path = os.path.join(image_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=150)
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]

        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1].strip()

        txt_name = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(caption_dir, txt_name), "w", encoding="utf-8") as f:
            f.write(caption)

    except Exception as e:
        print(f"Lỗi với {img_file}: {e}")

