# OminiControl

## Train
Training with noise, adding on conditional image input(spatial aligned tasks).

## Eval 

### CLIP & FID
1. Load LoRA weights and generate 200 images from COCO validation set(save_eval_results.py). Outputs in eval_{conditional_type} dir.
2. Evaluate images from eval_{conditional_type} dir by eval_from_dir.py.