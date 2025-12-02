import torch
from PIL import Image
from transformers import AutoModel
from huggingface_hub import hf_hub_download
from torchvision.transforms import functional as F

def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    # Resize the image to have a fixed height of 64 pixels
    img = img.resize((img.width * 64 // img.height, 64))
    img = F.to_tensor(img)
    img = F.normalize(img, [0.5], [0.5])
    return img

# 1. Load the model
model = AutoModel.from_pretrained("blowing-up-groundhogs/emuru", trust_remote_code=True)
model.cuda()  # Move to GPU if available

# 2. Prepare your inputs
style_text = 'THE JOLLY IS "U"'
gen_text = 'EMURU'
img_path = hf_hub_download(repo_id="blowing-up-groundhogs/emuru", filename="sample.png")
style_img = load_image(img_path)
style_img = style_img.cuda()

# 3. Generate an image
generated_pil_image = model.generate(
    style_text=style_text,
    gen_text=gen_text,
    style_img=style_img,
    max_new_tokens=64
)

# 4. Save the result
generated_pil_image.save("generated_image.png")
