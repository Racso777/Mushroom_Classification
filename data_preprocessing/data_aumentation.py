import os
import cv2
import numpy as np

# 水平翻转
def Horizontal(img):
    return cv2.flip(img, 1, dst=None)  # dst=None is default and can be omitted

# 垂直翻转
def Vertical(img):
    return cv2.flip(img, 0, dst=None)  # dst=None is default and can be omitted

if __name__ == "__main__":
    from_root = "F:\mushroom_project\dataset"
    save_root = "enhance_dataset"
    threshold = 1000

    for a, b, c in os.walk(from_root):
        print(os.path.basename(a))
        for file_i in c:
            file_i_path = os.path.join(a, file_i)

            split = os.path.splitext(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_root, os.path.basename(a))
            #print(file_i_path)
            print(save_path)

            if os.path.isdir(save_path) == False:
                os.makedirs(save_path)

            img_i = cv2.imdecode(np.fromfile(file_i_path, dtype=np.uint8), -1)  # 读取图片

            #cv2.imencode('.jpg', img_i)[1].tofile(os.path.join(save_path, file_i[:-4] + "_original.jpg"))  # 保存原图

            img_horizontal = Horizontal(img_i)
            #cv2.imencode('.jpg', img_horizontal)[1].tofile(os.path.join(save_path, file_i[:-4] + "_horizontal.jpg"))  # 保存水平翻转图

            img_vertical = Vertical(img_i)
            #cv2.imencode('.jpg', img_vertical)[1].tofile(os.path.join(save_path, file_i[:-4] + "_vertical.jpg"))  # 保存垂直翻转图

