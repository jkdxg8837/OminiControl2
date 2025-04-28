import subprocess  # 你需要定义这个变量

command = [
    "python", "eval_from_dir.py",

]

subprocess.run(command, check=True)