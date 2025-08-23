import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import torchvision.utils as tvu
import numpy as np

# 图像转换：调整大小并转换为 tensor，范围归一化到 [0, 1]
transform = transforms.Compose([
    transforms.ToTensor()
])

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.clone()  # 防止修改原图像
    channels, height, width = image.shape

    # 添加盐噪声
    num_salt = int(height * width * salt_prob)
    for _ in range(num_salt):
        h = np.random.randint(0, height)  # 随机选择高度索引
        w = np.random.randint(0, width)   # 随机选择宽度索引
        noisy_image[:, h, w] = 1  # 将该位置所有通道设置为盐噪声（白）

    # 添加胡椒噪声
    num_pepper = int(height * width * pepper_prob)
    for _ in range(num_pepper):
        h = np.random.randint(0, height)  # 随机选择高度索引
        w = np.random.randint(0, width)   # 随机选择宽度索引
        noisy_image[:, h, w] = 0  # 将该位置所有通道设置为胡椒噪声（黑）

    noisy_image = torch.clamp(noisy_image, 0, 1)  # 确保像素值限制在0到1之间
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob, mask_folder=None, filename=None):
    noisy_image = image.clone()  # 防止修改原图像
    channels, height, width = image.shape

    # 创建一个全零的mask，用于标记椒盐噪声位置
    mask = torch.ones((height, width), dtype=torch.uint8)

    # 添加盐噪声
    num_salt = int(height * width * salt_prob)
    for _ in range(num_salt):
        h = np.random.randint(0, height)  # 随机选择高度索引
        w = np.random.randint(0, width)   # 随机选择宽度索引
        noisy_image[:, h, w] = 1  # 将该位置所有通道设置为盐噪声（白）
        mask[h, w] = 0  # 标记为噪声点

    # 添加胡椒噪声
    num_pepper = int(height * width * pepper_prob)
    for _ in range(num_pepper):
        h = np.random.randint(0, height)  # 随机选择高度索引
        w = np.random.randint(0, width)   # 随机选择宽度索引
        noisy_image[:, h, w] = 0  # 将该位置所有通道设置为胡椒噪声（黑）
        mask[h, w] = 0  # 标记为噪声点

    noisy_image = torch.clamp(noisy_image, 0, 1)  # 确保像素值限制在0到1之间

    # 如果提供了mask_folder和filename，则保存mask为npy文件
    if mask_folder and filename:
        os.makedirs(mask_folder, exist_ok=True)
        mask_path = os.path.join(mask_folder, os.path.splitext(filename)[0] + "_mask.npy")
        np.save(mask_path, mask.numpy())  # 保存为npy文件，值是0或1
        print(f"Saved mask to {mask_path}")

    return noisy_image


# todo 添加泊松噪声
def add_poisson_noise(tensor_image):
    # 生成泊松噪声
    # tensor_image假设已经在0到1范围内
    noisy_image = torch.poisson(tensor_image * 255) / 255.0
    # 确保输出仍在0到1之间
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

def add_poisson_noise_intensity(tensor_image, intensity=2.0):
    """
    添加泊松噪声
    :param tensor_image: 输入图像，假设在 0 到 1 范围内
    :param intensity: 噪声强度，>1 增强噪声，<1 减弱噪声
    :return: 添加噪声后的图像
    """
    # 调整 λ 参数
    scaled_image = tensor_image * intensity * 255
    noisy_image = torch.poisson(scaled_image) / (intensity * 255.0)
    # 确保输出仍在 0 到 1 之间
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

# todo 添加高斯噪声
def add_gau_noise(x, sigma=0.1):
    x = x + sigma * torch.randn_like(x)
    x = torch.clamp(x, 0.0, 1.0)  # todo not clamp
    return x

# todo 添加瑞丽噪声
def add_ray_noise(x, sigma_r = 0.2):
    # 生成瑞利噪声
    rayleigh_noise = sigma_r * torch.sqrt(-2 * torch.log(1 - torch.rand_like(x)))
    mean_noise = torch.mean(rayleigh_noise)  # todo avoid 亮度变化
    # 将噪声添加到图像上
    x = x + rayleigh_noise - mean_noise
    # 使用 clamp 确保范围在 [0, 1] 之间
    x = torch.clamp(x, 0.0, 1.0)
    return x

def add_uni_noise(x):
    # todo 设置均匀噪声的范围，a 为下限，b 为上限
    a = -0.1  # 噪声的下限
    b = 0.1  # 噪声的上限
    # 生成均匀噪声
    uniform_noise = (b - a) * torch.rand_like(x) + a  # 均匀分布噪声
    # 将噪声添加到图像上
    x = x + uniform_noise
    # 使用 clamp 确保范围在 [0, 1] 之间
    x = torch.clamp(x, 0.0, 1.0)
    return x



base_path = "/media/zyserver/data16t/lpd/ddrm_512/images"


path_pairs = [
    # todo 添加高斯噪声1
    {
        "input_folder": f"{base_path}/input",
        "output_folder": f"{base_path}/source_1_gau_0.1",
        "process": lambda x: add_gau_noise(x, sigma=0.1),  # 高斯噪声
    },

    # todo 添加椒盐噪声1
    {
        "input_folder": f"{base_path}/input",
        "output_folder": f"{base_path}/source_1_impulse_0.04",
        "mask_folder": f"{base_path}/source_1_impulse_0.04_mask",
        # "process": lambda x: add_salt_and_pepper_noise(x, salt_prob=0.02, pepper_prob=0.02),  # 椒盐噪声
        "process": lambda x, mask_folder, filename: add_salt_and_pepper_noise(x, salt_prob=0.02, pepper_prob=0.02, mask_folder=mask_folder, filename=filename),
    },
    
    # todo 添加泊松噪声1
    {
        "input_folder": f"{base_path}/input",
        "output_folder": f"{base_path}/source_1_poisson",
        "process": lambda x: add_poisson_noise(x),  # 椒盐噪声
    },

    # todo 添加瑞利噪声1
    {
        "input_folder": f"{base_path}/input",
        "output_folder": f"{base_path}/source_1_ray_0.2",
        "process": lambda x: add_ray_noise(x, 0.2),  # 瑞利噪声
    },

    # todo 添加均匀噪声1
    {
        "input_folder": f"{base_path}/input",
        "output_folder": f"{base_path}/source_1_uni",
        "process": lambda x: add_uni_noise(x),  # 均匀噪声
    },
    
]

# 遍历路径对并处理
for paths in path_pairs:
    input_folder = paths["input_folder"]
    output_folder = paths["output_folder"]
    process = paths["process"]
    mask_folder = paths.get("mask_folder")  # 获取mask_folder（如果存在）

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的每个文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 根据需要处理的图像格式调整
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert("RGB")

            # todo 转换为 tensor 并归一化
            x = transform(image)

            # 执行对应的处理逻辑，传递mask_folder和filename（如果适用）
            if mask_folder:
                os.makedirs(mask_folder, exist_ok=True)
                x = process(x, mask_folder=mask_folder, filename=filename)
            else:
                x = process(x)
                
            # # todo 执行对应的处理逻辑
            # x = process(x)

            x_noisy_image = transforms.ToPILImage()(x)  # todo 将张量转到transforms前，确保值x归一化到正确的范围
            # 保存加噪声后的图像
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")  # 保存时使用原文件名
            x_noisy_image.save(output_path, 'PNG')

            print(f"Processed {filename} and saved to {output_path}")
