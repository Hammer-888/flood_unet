import os
import shutil
import random

file_path = "train_copy"

target_path = "data"

vv_list = os.listdir(os.path.join(file_path, "vv"))
vh_list = os.listdir(os.path.join(file_path, "vh"))
label_list = os.listdir(os.path.join(file_path, "label"))


random.shuffle(label_list)


label_train = label_list[: int(len(label_list) * 0.8)]

label_val = label_list[int(len(label_list) * 0.8) :]


for i in label_train:
    i_name = i.split(".")[0]
    os.makedirs(os.path.join(target_path, "train", "label"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "train", "vv"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "train", "vh"), exist_ok=True)
    shutil.copy(
        os.path.join(file_path, "label", i),
        os.path.join(target_path, "train", "label", i),
    )
    shutil.copy(
        os.path.join(file_path, "vv", os.path.join(i_name + "_vv" + ".png")),
        os.path.join(target_path, "train", "vv", os.path.join(i_name + "_vv" + ".png")),
    )
    shutil.copy(
        os.path.join(file_path, "vh", os.path.join(i_name + "_vh" + ".png")),
        os.path.join(target_path, "train", "vh", os.path.join(i_name + "_vh" + ".png")),
    )

for i in label_val:
    i_name = i.split(".")[0]
    os.makedirs(os.path.join(target_path, "val", "label"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "val", "vv"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "val", "vh"), exist_ok=True)
    shutil.copy(
        os.path.join(file_path, "label", i),
        os.path.join(target_path, "val", "label", i),
    )
    shutil.copy(
        os.path.join(file_path, "vv", os.path.join(i_name + "_vv" + ".png")),
        os.path.join(target_path, "val", "vv", os.path.join(i_name + "_vv" + ".png")),
    )
    shutil.copy(
        os.path.join(file_path, "vh", os.path.join(i_name + "_vh" + ".png")),
        os.path.join(target_path, "val", "vh", os.path.join(i_name + "_vh" + ".png")),
    )
