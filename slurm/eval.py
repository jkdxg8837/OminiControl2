import subprocess  # 你需要定义这个变量

command = [
    "python", "save_eval_results.py",
    "--path1", "./runs/canny/3",
    "--path2", "./ckpt/eval/subject.safetensors",
]

subprocess.run(command, check=True)