import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image


def get_gaussian_kernel2d(kernel_size=5, sigma=1.0, channels=3):
    """生成可用于卷积的2D高斯核(每通道相同)"""
    # 创建坐标网格
    ax = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # 扩展成适配 group conv2d 的卷积核格式
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)  # shape: [C, 1, k, k]
    return kernel


def gaussian_blur(y, kernel_size=5, sigma=1.0):
    """对输入张量 y 进行高斯滤波"""
    channels = y.shape[1]
    kernel = get_gaussian_kernel2d(kernel_size, sigma, channels).to(y.device)
    padding = kernel_size // 2
    y_blur = F.conv2d(y, kernel, padding=padding, groups=channels)
    return y_blur

def load_image_tensor(path, device='cpu'):
    """
    从路径加载图像并转换为 [0,1] 的 torch.Tensor,格式为 (1, C, H, W)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,255] -> [0,1] and (H,W,C) -> (C,H,W)
    ])
    image = Image.open(path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    return img_tensor, image

def estimate_sigma_from_y(y_tensor, kernel_size=5): # kernel_size越大 平滑程度越高.
    # print ("-------------------------y_tensor.shape = ", y_tensor.shape)
    y_blur = F.avg_pool2d(y_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    # y_blur = gaussian_blur(y_tensor, kernel_size=kernel_size, sigma=1.0)
    # print ("--------------------------------type = ", type(y_blur))
    save_image(y_blur, "y_blur.png")
    noise = y_tensor - y_blur
    save_image(noise, "noise.png")
    sigma_eq = torch.std(noise)
    return sigma_eq.item(), noise


  # 每个像素的平方，近似它的方差


def estimate_sigma(y_tensor, kernel_size=5): # kernel_size越大 平滑程度越高.
    # print ("-------------------------y_tensor.shape = ", y_tensor.shape)
    y_blur = F.avg_pool2d(y_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    # y_blur = gaussian_blur(y_tensor, kernel_size=kernel_size, sigma=1.0)
    # print ("--------------------------------type = ", type(y_blur))
    save_image(y_blur, "y_blur.png")
    noise = y_tensor - y_blur
    save_image(noise, "noise.png")
    sigma_eq = torch.std(noise)
    return sigma_eq.item()


def estimate_sigma_local(y_tensor, kernel_size=5): # kernel_size越大 平滑程度越高.
    y_blur = F.avg_pool2d(y_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    save_image(y_blur, "y_blur.png")
    noise = y_tensor - y_blur
    # noise_var_map = noise ** 2 
    # noise_var_map = torch.sqrt(torch.abs(noise) # torch.abs(noise)
    noise_var_map = torch.abs(noise)
    # noise_var_map = noise_var_map / noise_var_map.max()  # 归一化到 [0,1]
    print("-----------------------noise_var_map.shape = ", noise_var_map.shape)
    save_image(noise_var_map, "noise_var.png") 
    return noise_var_map

# def estimate_sigma_local(y_tensor, kernel_size=5):
#     """
#     返回每个像素的局部噪声标准差 (sigma map)
#     y_tensor: [B,C,H,W]
#     """
#     # 先平滑得到近似干净图像
#     y_blur = F.avg_pool2d(y_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

#     # 残差噪声
#     noise = y_tensor - y_blur

#     # 残差平方
#     noise_sq = noise ** 2

#     # 在局部窗口内求均值，相当于局部方差估计
#     local_var = F.avg_pool2d(noise_sq, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

#     # 标准差 = sqrt(方差)
#     local_sigma = torch.sqrt(local_var + 1e-8)
#     save_image(local_sigma, "local_sigma.png") 
#     return local_sigma  # [B,C,H,W] 每个像素的局部sigma

def local_variance_map(noise, kernel_size=5):
    # 局部均值
    mean_local = F.avg_pool2d(noise, kernel_size, stride=1, padding=kernel_size//2)
    # 局部平方均值
    mean_sq_local = F.avg_pool2d(noise**2, kernel_size, stride=1, padding=kernel_size//2)
    # 方差 = E[x^2] - (E[x])^2
    var_local = mean_sq_local - mean_local**2
    save_image(var_local, "var_local.png") 
    return var_local

def estimate_impulse_noise_level(y, threshold_low=0.02, threshold_high=0.98):
    # 统计接近0和1的像素点
    salt_pixels = (y >= threshold_high).float()
    pepper_pixels = (y <= threshold_low).float()
    
    # 计算椒盐噪声像素比例
    total_pixels = y.numel()
    salt_ratio = salt_pixels.sum() / total_pixels
    pepper_ratio = pepper_pixels.sum() / total_pixels
    
    # 总椒盐噪声比例
    impulse_ratio = salt_ratio + pepper_ratio
    
    # 将比例转换为噪声水平 (可以根据需要调整映射函数)
    # impulse_level = torch.sqrt(impulse_ratio) * 0.5  # 经验公式
    impulse_level = impulse_ratio
    
    return impulse_level

if __name__ == "__main__":
    # 设置路径
    # image_path = "/media/zyserver/data16t/lpd/ddrm_512/images_256/00055_00_256.png"
    image_path = "/media/zyserver/data16t/lpd/ddrm_512/images/source_1_impulse_0.04/00055_00_256.png"

    # 加载图像
    y_tensor, y_pil = load_image_tensor(image_path)

    # 估计 σ
    # sigma_eq, noise_tensor = estimate_sigma_from_y(y_tensor, kernel_size=5)
    # print(f"Estimated equivalent Gaussian noise sigma: {(sigma_eq * 1.5):.4f}")
    local_variance_map(y_tensor)
    estimate_sigma_local(y_tensor)
