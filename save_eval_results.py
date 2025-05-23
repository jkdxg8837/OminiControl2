import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.flux.generate import generate
from src.flux.condition import Condition
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw
import torchvision.transforms as T
import yaml
from src.train.model import OminiModel
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path1", type=str, default="")
parser.add_argument("--path2", type=str, default="")
parser.add_argument("--val_length", type=int, default=200)

args = parser.parse_args()
print(f"evaluating by using ckpt:{args.path1 + args.path2}")

path1 = args.path1
path2 = args.path2
val_length = args.val_length

with open(os.path.join(path1, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)
condition_size = config['train']['dataset']['condition_size']
target_size = config['train']['dataset']['target_size']
position_scale = config['train']['dataset']['position_scale']
condition_type = config['train']['condition_type']
to_tensor = T.ToTensor()
# Load COCO Val dataset from huggingface
dataset = load_dataset("jxie/coco_captions", split="validation")[:200]
image_list = dataset["image"]
prompt_list = dataset["caption"]
save_path = f"./eval_{condition_type}"
@torch.no_grad()
def generate_a_sample(
    dataset,
    # pl_module,
    save_path,
    condition_type="super_resolution",
):
    # TODO: change this two variables to parameters

    generator = torch.Generator(device=torch.device("cuda"))
    generator.manual_seed(42)

    test_list = []

    if condition_type == "subject":
        test_list.extend(
            [
                (
                    Image.open("assets/test_in.jpg"),
                    [0, -32],
                    "Resting on the picnic table at a lakeside campsite, it's caught in the golden glow of early morning, with mist rising from the water and tall pines casting long shadows behind the scene.",
                ),
                (
                    Image.open("assets/test_out.jpg"),
                    [0, -32],
                    "In a bright room. It is placed on a table.",
                ),
            ]
        )
    elif condition_type == "canny":
        from tqdm import tqdm
        for i in tqdm(range(len(image_list))):
            prompt = prompt_list[i]
            condition_img = load_image(image_list[i]).resize((condition_size, condition_size))
            # condition_img = Image.open("assets/vase_hq.jpg").resize(
            #         (condition_size, condition_size)
            #     )
            condition_img = np.array(condition_img)
            condition_img = cv2.Canny(condition_img, 100, 200)
            condition_img = Image.fromarray(condition_img).convert("RGB")
            test_list.append(
                (
                    condition_img,
                    [0, 0],
                    prompt,
                    {"position_scale": position_scale} if position_scale != 1.0 else {},
                )
            )
            pass
    elif condition_type == "coloring":
        condition_img = (
            Image.open("assets/vase_hq.jpg")
            .resize((condition_size, condition_size))
            .convert("L")
            .convert("RGB")
        )
        test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
    elif condition_type == "depth":
        if not hasattr(self, "deepth_pipe"):
            self.deepth_pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device="cpu",
            )
        condition_img = (
            Image.open("assets/vase_hq.jpg")
            .resize((condition_size, condition_size))
            .convert("RGB")
        )
        condition_img = self.deepth_pipe(condition_img)["depth"].convert("RGB")
        test_list.append(
            (
                condition_img,
                [0, 0],
                "A beautiful vase on a table.",
                {"position_scale": position_scale} if position_scale != 1.0 else {},
            )
        )
    elif condition_type == "depth_pred":
        condition_img = (
            Image.open("assets/vase_hq.jpg")
            .resize((condition_size, condition_size))
            .convert("RGB")
        )
        test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
    elif condition_type == "deblurring":
        blur_radius = 5
        image = Image.open("./assets/vase_hq.jpg")
        condition_img = (
            image.convert("RGB")
            .resize((condition_size, condition_size))
            .filter(ImageFilter.GaussianBlur(blur_radius))
            .convert("RGB")
        )
        test_list.append(
            (
                condition_img,
                [0, 0],
                "A beautiful vase on a table.",
                {"position_scale": position_scale} if position_scale != 1.0 else {},
            )
        )
    elif condition_type == "fill":
        condition_img = (
            Image.open("./assets/vase_hq.jpg")
            .resize((condition_size, condition_size))
            .convert("RGB")
        )
        mask = Image.new("L", condition_img.size, 0)
        draw = ImageDraw.Draw(mask)
        a = condition_img.size[0] // 4
        b = a * 3
        draw.rectangle([a, a, b, b], fill=255)
        condition_img = Image.composite(
            condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
        )
        test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
    elif condition_type == "sr":
        condition_img = (
            Image.open("assets/vase_hq.jpg")
            .resize((condition_size, condition_size))
            .convert("RGB")
        )
        test_list.append((condition_img, [0, -16], "A beautiful vase on a table."))
    elif condition_type == "cartoon":
        condition_img = (
            Image.open("assets/cartoon_boy.png")
            .resize((condition_size, condition_size))
            .convert("RGB")
        )
        test_list.append(
            (
                condition_img,
                [0, -16],
                "A cartoon character in a white background. He is looking right, and running.",
            )
        )
    else:
        raise NotImplementedError
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return test_list


generated_images = []

test_list = generate_a_sample(dataset, f"./eval_{condition_type}", condition_type)

trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=config["train"]["lora_config"],
        adapter_names=[config["train"]["condition_type"]],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=config["train"]["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=config["train"].get("gradient_checkpointing", False),
    )


trainable_model.load_lora(path1, path2,[condition_type])
for i, (condition_img, position_delta, prompt, *others) in enumerate(test_list):
    condition = Condition(
        condition_type=condition_type,
        condition=condition_img.resize(
            (condition_size, condition_size)
        ).convert("RGB"),
        position_delta=position_delta,
        **(others[0] if others else {}),
    )
    generator = torch.Generator(device=torch.device("cuda"))
    generator.manual_seed(42)

    with torch.no_grad():
        res = generate(
            trainable_model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size,
            width=target_size,
            generator=generator,
            model_config=trainable_model.model_config,
            default_lora=True,
        )
    res.images[0].save(
        os.path.join(save_path, f"{condition_type}_{i}.jpg")
    )

from eval_metrics import compute_metrics
metrics = compute_metrics(val_length, f"./eval_{condition_type}", condition_type)