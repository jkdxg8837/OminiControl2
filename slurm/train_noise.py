import os
import subprocess

# 设置环境变量
os.environ['XFL_CONFIG'] = './train/config/omini2/canny_512_ct_fr.yaml'
os.environ['WANDB_API_KEY'] = '05e08cadfa8f29b4fb78407724e732c453648de6'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# 打印环境变量
print(os.environ['XFL_CONFIG'])

# 启动训练命令
subprocess.run([
    "accelerate", "launch",
    "--main_process_port", "24242",
    "-m", "src.train.train"
])
