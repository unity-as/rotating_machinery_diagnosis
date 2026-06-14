import subprocess
import sys

scripts = [
    "scripts/step1_extract_words.py",
    "scripts/step2_train_pvdbow.py",
    "scripts/step3_fuse_and_save.py",
    "scripts/step4_train_classifier.py"
]

for script in scripts:
    print(f"\n===== 运行 {script} =====")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"脚本 {script} 运行失败，停止。")
        break
else:
    print("全部流程完成！")
