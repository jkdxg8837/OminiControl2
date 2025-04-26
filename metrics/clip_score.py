from torchmetrics.functional.multimodal import clip_score
from functools import partial
# Load the images and prompts
parser = argparse.ArgumentParser()
parser.add_argument("--val_length", type=int, default=200)
parser.add_argument("--img_path", type=str, default="")
parser.add_argument("--condition_type", type=str, default="")

args = parser.parse_args()
print(f"evaluating length:{args.val_length}")
img_save_path = args.img_path
condition_type = args.condition_type
dataset_length = args.val_length

dataset = load_dataset("jxie/coco_captions", split="validation")[:dataset_length]
image_list = dataset["image"]
prompt_list = dataset["caption"]

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
generated_images = []
for i in tqdm(range(val_length)):
    img_path = os.path.join(img_save_path, condition_type+"_"+str(i)+".jpg")
    image = Image.open(img_path).convert("RGB")
    generated_images.append(image)
def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

sd_clip_score = calculate_clip_score(generated_images, prompt_list)
print(f"CLIP score: {sd_clip_score}")
