import os
import shutil

# 定义原始train目录和目标目录
train_dir = "train"
new_base_dir = "train_copy"

# 遍历train目录下的所有子目录
for root, dirs, files in os.walk(train_dir):
    if "tiles" in dirs:
        sub_dir_path = os.path.join(root, "tiles")

        vv_path = os.path.join(sub_dir_path, "vv")
        vh_path = os.path.join(sub_dir_path, "vh")
        flood_label_path = os.path.join(sub_dir_path, "flood_label")

        if (
            not os.path.exists(vv_path)
            or not os.path.exists(vh_path)
            or not os.path.exists(flood_label_path)
        ):
            continue

        for f in os.listdir(vv_path):
            if f.endswith(".png"):
                shutil.copy(
                    os.path.join(vv_path, f), os.path.join(new_base_dir, "vv", f)
                )

        for f in os.listdir(vh_path):
            if f.endswith(".png"):
                shutil.copy(
                    os.path.join(vh_path, f), os.path.join(new_base_dir, "vh", f)
                )

        for f in os.listdir(flood_label_path):
            if f.endswith(".png"):
                shutil.copy(
                    os.path.join(flood_label_path, f),
                    os.path.join(new_base_dir, "label", f),
                )

print("所有子目录文件复制并分割完成")
