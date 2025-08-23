import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
    """
    真正简化的像素空间版本 - 不使用任何奇异值分解概念
    """
    with torch.no_grad():
        # 简化的初始化 - 直接在像素空间
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        
        # 对于去噪：如果largest_sigmas > sigma_0，主要用观测；否则混合
        if largest_sigmas[0, 0, 0, 0] > sigma_0:
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
            
            # 1. 首先应用missing pixels策略（对应etaC）
            xt_next = x0_t + sigma_tilde_nextC * et + std_nextC * torch.randn_like(x0_t)
            
            # 2. 如果sigma_next < sigma_0，覆盖为less noisy策略（对应etaA）
            if sigma_next < sigma_0:
                xt_next = x0_t + sigma_tilde_nextA * (y_0 - x0_t) / sigma_0 + std_nextA * torch.randn_like(x0_t)
            
            # 3. 如果sigma_next > sigma_0，覆盖为noisier策略（对应etaB）
            elif sigma_next > sigma_0:
                diff_sigma_t_nextB = torch.sqrt((sigma_next ** 2 - sigma_0 ** 2 * (etaB ** 2)).clamp(min=0))
                xt_next = etaB * y_0 + (1 - etaB) * x0_t + diff_sigma_t_nextB * torch.randn_like(x0_t)
            
            # 4. 应用alpha缩放
            xt_next = at_next.sqrt() * xt_next

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

# def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
#     """
#     修复版本：发现关键问题 - 原始代码中xt_mod_next计算方式的差异
#     """
#     with torch.no_grad():
#         # 初始化 x_T (保持原样)
#         largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
#         largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        
#         init_y = y_0 + largest_sigmas[0, 0, 0, 0] * x
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
            
#             # 预测 x0
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

#             # ===== 关键修复：按照原始代码的精确逻辑 =====
            
#             # 计算噪声水平
#             sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
#             sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            
#             # 关键：原始代码在这个"修正"空间中操作
#             xt_mod = xt / at.sqrt()[0, 0, 0, 0]
            
#             # 计算各种参数
#             std_nextC = sigma_next * etaC
#             sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)
            
#             std_nextA = sigma_next * etaA  
#             sigma_tilde_nextA = torch.sqrt(sigma_next ** 2 - std_nextA ** 2)
            
#             # 按照原始代码的逻辑：先处理所有像素，然后根据条件覆盖
            
#             # 1. 首先对所有像素应用missing pixels策略 (对应etaC)
#             Vt_xt_mod_next = x0_t + sigma_tilde_nextC * et + std_nextC * torch.randn_like(x0_t)
            
#             # 2. 根据条件覆盖像素
#             if sigma_next < sigma_0:
#                 # less noisy情况 - 使用etaA策略覆盖所有像素
#                 Vt_xt_mod_next = x0_t + sigma_tilde_nextA * (y_0 - x0_t) / sigma_0 + std_nextA * torch.randn_like(x0_t)
                
#             elif sigma_next > sigma_0:
#                 # noisier情况 - 使用etaB策略覆盖所有像素
#                 diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 * (etaB ** 2))
#                 Vt_xt_mod_next = y_0 * etaB + x0_t * (1 - etaB) + diff_sigma_t_nextB * torch.randn_like(x0_t)
            
#             # 3. 关键步骤：对应原始代码中的 H_funcs.V(Vt_xt_mod_next)
#             # 对于Denoising，这是恒等变换，但要保持一致的计算
#             xt_mod_next = Vt_xt_mod_next
            
#             # 4. 最终缩放 (对应原始代码的最后一步)
#             xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

#             x0_preds.append(x0_t.to('cpu'))
#             xs.append(xt_next.to('cpu'))

#     return xs, x0_preds



# def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
#     print("------------------------------x.shape = ", x.shape, "------y_0.shape = ", y_0.shape)
#     """
#     简化版本的去噪算法，去除了奇异值分解部分，直接在像素空间运行
#     """
#     with torch.no_grad():
#         # 初始化 x_T
#         largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
#         largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
#         print("------------------------largest_sigmas = ", largest_sigmas)
#         # 对于去噪任务，直接在像素空间初始化
#         # 使用简单的线性组合而不是复杂的奇异值分解操作
#         print("-------------------------------largest_sigmas.shape = ", largest_sigmas.shape)
#         print("-------------------------------x.shape = ", x.shape)
#         init_y = y_0 + largest_sigmas[0, 0, 0, 0] * x
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
            
#             # 预测 x0
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

#             # 简化的变分推断，直接在像素空间
#             sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
#             sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            
#             # 简化的更新规则，去除奇异值分解的复杂条件判断
#             if sigma_next > sigma_0:
#                 # 噪声较大的情况，使用 etaB 策略
#                 std_next = sigma_next * etaB
#                 sigma_tilde_next = torch.sqrt(sigma_next ** 2 - std_next ** 2)
                
#                 # 直接在像素空间混合观测数据和预测
#                 data_term = (y_0 - x0_t) / sigma_0
#                 xt_next = x0_t + sigma_tilde_next * data_term + std_next * torch.randn_like(x0_t)
                
#             else:
#                 # 噪声较小的情况，使用 etaA 策略  
#                 std_next = sigma_next * etaA
#                 sigma_tilde_next = torch.sqrt(sigma_next ** 2 - std_next ** 2)
                
#                 # 更强的数据约束
#                 data_term = (y_0 - x0_t) / sigma_0
#                 xt_next = x0_t + sigma_tilde_next * data_term + std_next * torch.randn_like(x0_t)
            
#             # 应用 alpha 缩放
#             xt_next = at_next.sqrt() * xt_next

#             x0_preds.append(x0_t.to('cpu'))
#             xs.append(xt_next.to('cpu'))

#     return xs, x0_preds































































