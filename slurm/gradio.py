import subprocess  # 你需要定义这个变量

command = [
    "python", "-m", "src.gradio.gradio_app",
]

subprocess.run(command, check=True)