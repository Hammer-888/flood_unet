import torch
import cv2
import os
import shutil
from network.model import load_network
from utils.config import default_argument_parser


def infer(args, lable_name):
    # load network
    network = load_network(
        args, f"outputs/flood-experiment/{args.struct}-{args.attention}.pth"
    )
    network.to("cuda")
    # 加载图像，手动指定图像路径

    label_path = "data/train/label/" + lable_name
    image_label = cv2.imread(label_path, 0) / 255.0
    image_vv = (
        cv2.imread(label_path.replace("label", "vv").replace(".png", "_vv.png"), 0)
        / 255.0
    ).astype("float32")
    image_vh = (
        cv2.imread(label_path.replace("label", "vh").replace(".png", "_vh.png"), 0)
        / 255.0
    ).astype("float32")

    # infer
    image_vv = torch.tensor(image_vv).unsqueeze(0).unsqueeze(0).cuda()
    image_vh = torch.tensor(image_vh).unsqueeze(0).unsqueeze(0).cuda()
    reslut = network(image_vv, image_vh)
    reslut = torch.sigmoid(reslut)
    reslut = reslut.cpu().detach().numpy().squeeze() * 255.0
    # 保存结果
    outpath = f"result/{args.struct}-{args.attention}-{lable_name}/"
    os.makedirs(outpath, exist_ok=True)
    shutil.copy(label_path, outpath + lable_name)
    shutil.copy(
        label_path.replace("label", "vv").replace(".png", "_vv.png"),
        outpath + lable_name.replace("label", "vv").replace(".png", "_vv.png"),
    )
    shutil.copy(
        label_path.replace("label", "vh").replace(".png", "_vh.png"),
        outpath + lable_name.replace("label", "vh").replace(".png", "_vh.png"),
    )
    cv2.imwrite(f"{outpath}{args.struct}-{args.attention}-{lable_name}", reslut)


if __name__ == "__main__":
    lable_name = "bangladesh_20170314t115609_x-5_y-28.png"
    parser = default_argument_parser()
    args = parser.parse_args()
    infer(args, lable_name)

# pic 1 "bangladesh_20170314t115609_x-5_y-28.png"
# pic 2 "bangladesh_20170314t115609_x-5_y-32.png"
# pic 3 "bangladesh_20170314t115609_x-11_y-21.png"
# pic 4 "bangladesh_20170314t115609_x-5_y-34.png"
# pic 5 "bangladesh_20170314t115609_x-5_y-33.png"
# pic 6 "bangladesh_20170314t115609_x-32_y-20.png"
