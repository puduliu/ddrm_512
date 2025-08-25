import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, sigma_0_map, etaB, etaA, etaC, cls_fn=None, classes=None):
    with torch.no_grad():
        print ("-------------------------------sigma_0.shape = ", sigma_0)
        print ("-------------------------------sigma_0_map.shape = ", sigma_0_map.shape)
        # sigma_0_map[:] = 0.7
        # 简化的初始化 - 直接在像素空间
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        
        # 对于去噪：如果largest_sigmas > sigma_0_map，主要用观测；否则混合
        if largest_sigmas[0, 0, 0, 0] > sigma_0:  # 保持原来的标量比较用于初始化
            # 观测的权重
            init_y = (sigma_0 / largest_sigmas[0, 0, 0, 0]) * y_0 + torch.sqrt(1 - (sigma_0 / largest_sigmas[0, 0, 0, 0])**2) * x
        else:
            init_y = y_0 + largest_sigmas[0, 0, 0, 0] * x
            
        x = init_y / largest_alphas.sqrt()
        
        # 设置迭代变量
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        # 在时间步上迭代
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            
            # 预测噪声
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]
            
            # 预测x0
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # 计算噪声水平
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            
            # 计算各种标准差
            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt((sigma_next ** 2 - std_nextC ** 2).clamp(min=0))
            
            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt((sigma_next ** 2 - std_nextA ** 2).clamp(min=0))
            
            # 模拟原始代码的三步逻辑，但完全在像素空间
            
            # 1. 首先应用missing pixels策略（对应etaC）- 作为默认策略
            xt_next = x0_t + sigma_tilde_nextC * et + std_nextC * torch.randn_like(x0_t)
            
            # 2. 创建逐像素比较的mask
            # sigma_0_map的形状: [1, 3, 256, 256]
            # 确保sigma_0_map在正确的设备上
            sigma_0_map_device = sigma_0_map.to(x.device)
            
            # 创建mask：哪些像素position满足 sigma_next < sigma_0_map
            less_noisy_mask = sigma_next < sigma_0_map_device  # [1, 3, 256, 256]
            
            # 创建mask：哪些像素position满足 sigma_next > sigma_0_map  
            noisier_mask = sigma_next > sigma_0_map_device  # [1, 3, 256, 256]
            
            # 2. 对于 sigma_next < sigma_0_map 的像素，应用less noisy策略（对应etaA）
            less_noisy_result = x0_t + sigma_tilde_nextA * (y_0 - x0_t) / sigma_0_map_device + std_nextA * torch.randn_like(x0_t)
            
            # 使用mask更新对应像素
            xt_next = torch.where(less_noisy_mask, less_noisy_result, xt_next)
            
            # 3. 对于 sigma_next > sigma_0_map 的像素，应用noisier策略（对应etaB）
            diff_sigma_t_nextB = torch.sqrt((sigma_next ** 2 - sigma_0_map_device ** 2 * (etaB ** 2)).clamp(min=0))
            noisier_result = etaB * y_0 + (1 - etaB) * x0_t + diff_sigma_t_nextB * torch.randn_like(x0_t)
            
            # 使用mask更新对应像素
            xt_next = torch.where(noisier_mask, noisier_result, xt_next)
            
            # 4. 应用alpha缩放
            xt_next = at_next.sqrt() * xt_next

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

# def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, sigma_0_map, etaB, etaA, etaC, cls_fn=None, classes=None):
#     with torch.no_grad():
#         print ("-------------------------------sigma_0_map.shape = ", sigma_0_map.shape)
#         # 简化的初始化 - 直接在像素空间
#         largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
#         largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        
#         # 对于去噪：如果largest_sigmas > sigma_0，主要用观测；否则混合
#         if largest_sigmas[0, 0, 0, 0] > sigma_0:
#             # 观测的权重
#             init_y = (sigma_0 / largest_sigmas[0, 0, 0, 0]) * y_0 + torch.sqrt(1 - (sigma_0 / largest_sigmas[0, 0, 0, 0])**2) * x
#         else:
#             init_y = y_0 + largest_sigmas[0, 0, 0, 0] * x
            
#         x = init_y / largest_alphas.sqrt()
        
#         # 设置迭代变量
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         x0_preds = []
#         xs = [x]

#         # 在时间步上迭代
#         for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(b, t.long())
#             at_next = compute_alpha(b, next_t.long())
#             xt = xs[-1].to('cuda')
            
#             # 预测噪声
#             if cls_fn == None:
#                 et = model(xt, t)
#             else:
#                 et = model(xt, t, classes)
#                 et = et[:, :3]
#                 et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
#             if et.size(1) == 6:
#                 et = et[:, :3]
            
#             # 预测x0
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

#             # 计算噪声水平
#             sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
#             sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            
#             # 计算各种标准差
#             std_nextC = sigma_next * etaC
#             sigma_tilde_nextC = torch.sqrt((sigma_next ** 2 - std_nextC ** 2).clamp(min=0))
            
#             std_nextA = sigma_next * etaA
#             sigma_tilde_nextA = torch.sqrt((sigma_next ** 2 - std_nextA ** 2).clamp(min=0))
            
#             # 模拟原始代码的三步逻辑，但完全在像素空间
            
#             # 1. 首先应用missing pixels策略（对应etaC）
#             xt_next = x0_t + sigma_tilde_nextC * et + std_nextC * torch.randn_like(x0_t)
            
#             # 2. 如果sigma_next < sigma_0，覆盖为less noisy策略（对应etaA）
#             if sigma_next < sigma_0:
#                 xt_next = x0_t + sigma_tilde_nextA * (y_0 - x0_t) / sigma_0 + std_nextA * torch.randn_like(x0_t)
            
#             # 3. 如果sigma_next > sigma_0，覆盖为noisier策略（对应etaB）
#             elif sigma_next > sigma_0:
#                 diff_sigma_t_nextB = torch.sqrt((sigma_next ** 2 - sigma_0 ** 2 * (etaB ** 2)).clamp(min=0))
#                 xt_next = etaB * y_0 + (1 - etaB) * x0_t + diff_sigma_t_nextB * torch.randn_like(x0_t)
            
#             # 4. 应用alpha缩放
#             xt_next = at_next.sqrt() * xt_next

#             x0_preds.append(x0_t.to('cpu'))
#             xs.append(xt_next.to('cpu'))

#     return xs, x0_preds

# def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
#     """
#     真正简化的像素空间版本 - 不使用任何奇异值分解概念
#     """
#     with torch.no_grad():
#         # 简化的初始化 - 直接在像素空间
#         largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
#         largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        
#         # 对于去噪：如果largest_sigmas > sigma_0，主要用观测；否则混合
#         if largest_sigmas[0, 0, 0, 0] > sigma_0:
#             # 观测的权重
#             init_y = (sigma_0 / largest_sigmas[0, 0, 0, 0]) * y_0 + torch.sqrt(1 - (sigma_0 / largest_sigmas[0, 0, 0, 0])**2) * x
#         else:
#             init_y = y_0 + largest_sigmas[0, 0, 0, 0] * x
            
#         x = init_y / largest_alphas.sqrt()
        
#         # 设置迭代变量
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         x0_preds = []
#         xs = [x]

#         # 在时间步上迭代
#         for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(b, t.long())
#             at_next = compute_alpha(b, next_t.long())
#             xt = xs[-1].to('cuda')
            
#             # 预测噪声
#             if cls_fn == None:
#                 et = model(xt, t)
#             else:
#                 et = model(xt, t, classes)
#                 et = et[:, :3]
#                 et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
#             if et.size(1) == 6:
#                 et = et[:, :3]
            
#             # 预测x0
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

#             # 计算噪声水平
#             sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
#             sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            
#             # 计算各种标准差
#             std_nextC = sigma_next * etaC
#             sigma_tilde_nextC = torch.sqrt((sigma_next ** 2 - std_nextC ** 2).clamp(min=0))
            
#             std_nextA = sigma_next * etaA
#             sigma_tilde_nextA = torch.sqrt((sigma_next ** 2 - std_nextA ** 2).clamp(min=0))
            
#             # 模拟原始代码的三步逻辑，但完全在像素空间
            
#             # 1. 首先应用missing pixels策略（对应etaC）
#             xt_next = x0_t + sigma_tilde_nextC * et + std_nextC * torch.randn_like(x0_t)
            
#             # 2. 如果sigma_next < sigma_0，覆盖为less noisy策略（对应etaA）
#             if sigma_next < sigma_0:
#                 xt_next = x0_t + sigma_tilde_nextA * (y_0 - x0_t) / sigma_0 + std_nextA * torch.randn_like(x0_t)
            
#             # 3. 如果sigma_next > sigma_0，覆盖为noisier策略（对应etaB）
#             elif sigma_next > sigma_0:
#                 diff_sigma_t_nextB = torch.sqrt((sigma_next ** 2 - sigma_0 ** 2 * (etaB ** 2)).clamp(min=0))
#                 xt_next = etaB * y_0 + (1 - etaB) * x0_t + diff_sigma_t_nextB * torch.randn_like(x0_t)
            
#             # 4. 应用alpha缩放
#             xt_next = at_next.sqrt() * xt_next

#             x0_preds.append(x0_t.to('cpu'))
#             xs.append(xt_next.to('cpu'))

#     return xs, x0_preds

































































