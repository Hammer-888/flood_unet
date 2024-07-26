import cv2
import os

label_path = r'data\val\label'

# 遍历label文件夹下所有图片，用cv2.imread打开,如果打开图像的整张像素为0，则删除这张图片
for root, dirs, files in os.walk(label_path):
    for file in files:
        path = os.path.join(root, file)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image.sum() == 0 or image.sum() == 255:
            os.remove(path)
            # 删除vv文件
            vv_path = path.replace('label', 'vv')
            vv_path = vv_path.replace('.png', '_vv.png')
            os.remove(vv_path)
            # 删除vh文件
            vh_path = path.replace('label', 'vh')
            vh_path = vh_path.replace('.png', '_vh.png')
            os.remove(vh_path)
print('done')