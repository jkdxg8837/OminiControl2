import subprocess  # 你需要定义这个变量

command = [
    "python", "save_eval_results.py",
]

subprocess.run(command, check=True)