import os
import subprocess
import sys
from pathlib import Path

def run():
    print("🚀 Starting all CIFAR-10 CNN experiments...")
    print("This will run 6 experiments sequentially. Estimated time: 1-2 hours on CPU.\n")
    
    # 确保在正确目录
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # 设置环境变量加速
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # 实时输出
    
    # 运行main.py
    try:
        subprocess.run([sys.executable, 'main.py'], env=env, check=True)
        print("\n✅ All experiments completed successfully!")
        print("📊 Check the results in the './results' folder")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during experiments: {e}")
        sys.exit(1)

if __name__ == '__main__':
    run()