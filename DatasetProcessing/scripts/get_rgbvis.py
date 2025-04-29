import sys
sys.path.append(r".")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
from src.utils.email_utils import send_email

# # 显示gray图像
# # cv2.imshow()

class RGBVIS:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        # 转换为浮点数
        self.img = self.img.astype(np.float32)
        R, G, B = cv2.split(self.img)
        bgr_sum = B + G + R + 1e-6
        self.b = B / bgr_sum
        self.g = G / bgr_sum
        self.r = R / bgr_sum
    
    def normalize_channel(self, channel):
        min_val = np.min(channel)
        max_val = np.max(channel)
        return (channel - min_val) / (max_val - min_val + 1e-6)
    
    def compute_EXG(self):
        return 2 * self.g - self.b - self.r
    
    def compute_EXR(self):
        return 1.4 * self.r - self.g
    
    def compute_EXGR(self):
        return self.compute_EXG() - 0.5 * self.r -  0.08 * self.b
    
    def compute_BI(self):
        return (1-self.r) / (self.r+self.g+self.b+1e-6)
    
    def compute_GRVI(self):
        return self.g / (self.r + 1e-6)
    
    def compute_NGRVI(self):
        return (self.g - self.r) / (self.g + self.r + 1e-6)
    
    def compute_MGRVI(self):
        return (self.g ** 2 - self.r ** 2) / (self.g ** 2 + self.r ** 2 + 1e-6)
    
    def compute_NGBVI(self):
        return (self.g - self.b) / (self.g + self.b + 1e-6)
    
    def compute_VDVI(self):
        return (2 * self.g - self.r - self.b) / (2 * self.g + self.r + self.b + 1e-6)
    
    def compute_RGRI(self):
        return self.r / (self.g + 1e-6)

    # 将所有的指数和rgb原图像合并为一个numpy数组中
    def stack(self):
        exg = self.normalize_channel(self.compute_EXG())
        exr = self.normalize_channel(self.compute_EXR())
        exgr = self.normalize_channel(self.compute_EXGR())
        bi = self.normalize_channel(self.compute_BI())
        grvi = self.normalize_channel(self.compute_GRVI())
        ngrvi = self.normalize_channel(self.compute_NGRVI())
        mgrvi = self.normalize_channel(self.compute_MGRVI())
        ngbvi = self.normalize_channel(self.compute_NGBVI())
        vdvi = self.normalize_channel(self.compute_VDVI())
        rgri = self.normalize_channel(self.compute_RGRI())
        return np.dstack((self.img / 255.0, exg, exr, exgr, bi, grvi, ngrvi, mgrvi, ngbvi, vdvi, rgri))

def load_and_show(npy_path):
    vis_stack = np.load(npy_path, allow_pickle=True)
    RGB =  vis_stack[:, :, :3]
    EXG = vis_stack[:, :, 3]
    EXR = vis_stack[:, :, 4]
    EXGR = vis_stack[:, :, 5]
    BI = vis_stack[:, :, 6]
    GRVI = vis_stack[:, :, 7]
    NGRVI = vis_stack[:, :, 8]
    MGRVI = vis_stack[:, :, 9]
    NGBVI = vis_stack[:, :, 10]
    VDVI = vis_stack[:, :, 11]
    RGRI = vis_stack[:, :, 12]

    # print(vis_stack.dtype)
    
    # 打印每种数据的最大最小值
    print("RGB max:", np.max(RGB), "min:", np.min(RGB))
    print("EXG max:", np.max(EXG), "min:", np.min(EXG))
    print("EXR max:", np.max(EXR), "min:", np.min(EXR))
    print("EXGR max:", np.max(EXGR), "min:", np.min(EXGR))
    print("BI max:", np.max(BI), "min:", np.min(BI))
    print("GRVI max:", np.max(GRVI), "min:", np.min(GRVI))
    print("NGRVI max:", np.max(NGRVI), "min:", np.min(NGRVI))
    print("MGRVI max:", np.max(MGRVI), "min:", np.min(MGRVI))
    print("NGBVI max:", np.max(NGBVI), "min:", np.min(NGBVI))
    print("VDVI max:", np.max(VDVI), "min:", np.min(VDVI))
    print("RGRI max:", np.max(RGRI), "min:", np.min(RGRI))
    # exit()
    
    
    plt.figure(figsize=(20, 10))
    font = {
    'weight' : 'normal',
    'size'   : 12,
    }

    plt.subplot(2,5,1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(EXG, cmap ='gray')
    plt.title('EXG', font)

    plt.subplot(2,5,2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(EXR, cmap ='gray')
    plt.title('EXR', font)

    plt.subplot(2,5,3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(EXGR, cmap ='gray')
    plt.title('EXGR', font)

    plt.subplot(2,5,4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(BI, cmap ='gray')
    plt.title('BI', font)

    plt.subplot(2,5,5)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(GRVI, cmap ='gray')
    plt.title('GRVI', font)

    plt.subplot(2,5,6)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(NGRVI, cmap ='gray')
    plt.title('NGRVI', font)

    plt.subplot(2,5,7)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(MGRVI, cmap ='gray')
    plt.title('MGRVI', font)

    plt.subplot(2,5,8)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(NGBVI, cmap ='gray')
    plt.title('NGBVI', font)

    plt.subplot(2,5,9)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(VDVI, cmap ='gray')
    plt.title('VDVI', font)

    plt.subplot(2,5,10)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(RGRI, cmap ='gray')
    plt.title('RGRI', font)

    Image.fromarray((RGB*255.0).astype(np.uint8)).show()

    # plt.imshow(BGR)
    plt.show()

    
    
if __name__ == "__main__":
    input_images_folder = r"E:/Code/RiceLodging/datasets/DJ/Lingtangkou/abnormal-03.30-7-640-0.1-0.6-0.2-0.2-v1/images"
    output_npy_folder = r"E:/Code/RiceLodging/datasets/DJ/Lingtangkou/vis"
    tasks = ['train', "val", "test"]
    for task in tasks:
        # 确保文件夹存在
        os.makedirs(os.path.join(output_npy_folder, task), exist_ok=True)
        images_list = os.listdir(os.path.join(input_images_folder, task))
        for image_name in tqdm(images_list, desc=task):
            if not image_name.endswith(".png"):
                continue
            rgb_vis = RGBVIS(os.path.join(input_images_folder, task, image_name))
            vis_stack = rgb_vis.stack()
            vis_stack.dump(os.path.join(output_npy_folder, task, image_name.replace('.png', '.npy')))
    
    send_email("RGBVIS Done!")

    # vis = RGBVIS("./GLCM/crop_test.tif")
    # vis_stack = vis.stack()
    # print(vis_stack.shape)
    # # 保存numpy数组
    # vis_stack.dump("./GLCM/crop_test_vis.npy")
    # load_and_show("./GLCM/crop_test_vis.npy")
    


