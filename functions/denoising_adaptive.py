import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a



# def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, sigma_0_map, etaB, etaA, etaC, cls_fn=None, classes=None):

#     with torch.no_grad():
        
#         sigma_0_map_flat = sigma_0_map.view(sigma_0_map.shape[0], -1)  # 修正：保持batch维度 [1, 196608]
#         print("----------------------sigma_0_map_flat.shape = ", sigma_0_map_flat.shape)
        
#         #setup vectors used in the algorithm
#         singulars = H_funcs.singulars()
#         print("----------------------singulars.shape = ", singulars.shape)
#         Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
#         Sigma[:singulars.shape[0]] = singulars
#         U_t_y = H_funcs.Ut(y_0)
#         Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

#         #initialize x_T as given in the paper
#         largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
#         largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt() # largest_sigmas = tensor([[[[97.1042]]]]) largest_sigmas[0,0,0,0] 就是取出 里面唯一的元素，得到一个 0维标量97.1042
#         print("----------------------largest_sigmas = ", largest_sigmas)
#         print("----------------------largest_sigmas.shape = ", largest_sigmas.shape)
        
#         large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0_map_flat.squeeze(0)) # 修正：使用squeeze(0)得到1D张量进行比较
#         print("----------------------large_singulars_index.type = ", type(large_singulars_index), "----len = ", len(large_singulars_index)) # .type =  <class 'tuple'> ----len =  1
#         inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device) #  inv_singulars_and_zero = torch.Size([196608])
#         print("-----------------inv_singulars_and_zero.shape = ", inv_singulars_and_zero.shape)

#         # vals = sigma_0_map_flat[large_singulars_index]   # shape = [k]
#         # nonzero_vals = vals[vals != 0]                   # 过滤掉 0
#         # print(nonzero_vals)
        
#         inv_singulars_and_zero[large_singulars_index] = sigma_0_map_flat.squeeze(0)[large_singulars_index] / singulars[large_singulars_index] # 修正：使用squeeze(0)
#         # sigma_0 / singulars[large_singulars_index] # sigma_0 小 → 观测质量好 sigma_0 大 → 观测质量差（相当于已经有噪声了）
#         inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

#         # implement p(x_T | x_0, y) as given in the paper
#         # if eigenvalue is too small, we just treat it as zero (only for init) 
#         init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
#         init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
#         init_y = init_y.view(*x.size())
#         remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
#         remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
#         init_y = init_y + remaining_s * x
#         init_y = init_y / largest_sigmas
        
#         #setup iteration variables
#         x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         x0_preds = []
#         xs = [x]

#         #iterate over the timesteps
#         for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(b, t.long())
#             at_next = compute_alpha(b, next_t.long())
#             xt = xs[-1].to('cuda')
#             if cls_fn == None:
#                 et = model(xt, t)
#             else:
#                 et = model(xt, t, classes)
#                 et = et[:, :3]
#                 et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
#             if et.size(1) == 6:
#                 et = et[:, :3]
            
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

#             #variational inference conditioned on y
#             sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
#             sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
#             # print("----------------------------sigma_next = ", sigma_next)
#             xt_mod = xt / at.sqrt()[0, 0, 0, 0]
#             V_t_x = H_funcs.Vt(xt_mod)
#             SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
#             V_t_x0 = H_funcs.Vt(x0_t)
#             SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

#             falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
#             cond_before_lite = singulars * sigma_next > sigma_0_map_flat.squeeze(0)  # 修正：使用squeeze(0)得到1D张量
#             print("============================cond_before_lite.shape = ", cond_before_lite.shape)
#             print("============================singulars.shape = ", singulars.shape)
#             cond_after_lite = singulars * sigma_next < sigma_0_map_flat.squeeze(0)   # 修正：使用squeeze(0)得到1D张量
#             cond_before = torch.hstack((cond_before_lite, falses))
#             cond_after = torch.hstack((cond_after_lite, falses))

#             std_nextC = sigma_next * etaC
#             sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

#             std_nextA = sigma_next * etaA
#             sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
            
#             diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0_map_flat.squeeze(0)[cond_before_lite] ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

#             #missing pixels
#             Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

#             #less noisy than y (after)
#             Vt_xt_mod_next[:, cond_after] = \
#                 V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0_map_flat[:, :U_t_y.shape[1]])[:, cond_after_lite[:U_t_y.shape[1]]] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            
#             #noisier than y (before)
#             # 修正：扩展diff_sigma_t_nextB的维度以匹配
#             diff_sigma_expanded = torch.zeros_like(U_t_y[:, cond_before_lite[:U_t_y.shape[1]]])
#             if cond_before_lite[:U_t_y.shape[1]].any():
#                 diff_sigma_expanded[0, :] = diff_sigma_t_nextB[:torch.sum(cond_before_lite[:U_t_y.shape[1]])]
            
#             Vt_xt_mod_next[:, cond_before] = \
#                 (Sig_inv_U_t_y[:, cond_before_lite[:U_t_y.shape[1]]] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_expanded * torch.randn_like(U_t_y)[:, cond_before_lite[:U_t_y.shape[1]]])

#             #aggregate all 3 cases and give next prediction
#             xt_mod_next = H_funcs.V(Vt_xt_mod_next)
#             xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

#             x0_preds.append(x0_t.to('cpu'))
#             xs.append(xt_next.to('cpu'))

#     return xs, x0_preds

import numpy as np
def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, sigma_0_map, etaB, etaA, etaC, cls_fn=None, classes=None):

    with torch.no_grad():
        sigma_0_map_flat = sigma_0_map.view(-1)  # 跟reshape元素顺序应该一样
        print("----------------------sigma_0_map_flat.shape = ", sigma_0_map_flat.shape)
        np.savetxt("sigma_0_map_flat.txt", sigma_0_map_flat.cpu().numpy(), fmt="%.2f")
        #setup vectors used in the algorithm
        singulars = H_funcs.singulars()
        print("----------------------singulars.shape = ", singulars.shape)
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt() # largest_sigmas = tensor([[[[97.1042]]]]) largest_sigmas[0,0,0,0] 就是取出 里面唯一的元素，得到一个 0维标量97.1042
        print("----------------------largest_sigmas = ", largest_sigmas)
        print("----------------------largest_sigmas.shape = ", largest_sigmas.shape)
        
        # sigma_0_map_flat[sigma_0_map_flat > 0.2] = 1 # TODO test impainting是直接让奇异值等于0了
        # print("----------------------------------sigma_0_map_flat = ", sigma_0_map_flat)
        # mask = sigma_0_map_flat > 0.2 
        # singulars[mask] = 0.05
        
        count = (sigma_0_map_flat > 0.2).sum()
        print("==========================count = ", count.item())  # 转成 Python 标量

        # large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0_map_flat) # singulars * largest_sigmas[0,0,0,0] 向量 * 标量，所以结果还是一个 shape 为 [196608] 的向量。
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0_map_flat) # TODO inpainting 形状会对不上. 因为我的假设去噪所有奇异值都是满的
     

        print("----------------------large_singulars_index.type = ", type(large_singulars_index), "----len = ", len(large_singulars_index)) # .type =  <class 'tuple'> ----len =  1
        print("------------------------------large_singulars_index = ", large_singulars_index)
        print("------------------------------large_singulars_index.shape = ", large_singulars_index[0].shape)
        print(large_singulars_index[0].numel())
        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device) #  inv_singulars_and_zero = torch.Size([196608])
        print("-----------------inv_singulars_and_zero.shape = ", inv_singulars_and_zero.shape)

        # vals = sigma_0_map_flat[large_singulars_index]   # shape = [k]
        # nonzero_vals = vals[vals != 0]                   # 过滤掉 0
        # print(nonzero_vals)
        
        inv_singulars_and_zero[large_singulars_index] = sigma_0_map_flat[large_singulars_index] / singulars[large_singulars_index] # singulars[large_singulars_index] -> 取出 singulars 中所有满足条件的位置的值，得到一个 1D tensor
        # sigma_0 / singulars[large_singulars_index] # sigma_0 小 → 观测质量好 sigma_0 大 → 观测质量差（相当于已经有噪声了）
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     
        print("----------------------------inv_singulars_and_zero = \n", inv_singulars_and_zero)

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init) 
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
        remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas
        
        #setup iteration variables
        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            #variational inference conditioned on y
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            # print("----------------------------sigma_next = ", sigma_next)
            xt_mod = xt / at.sqrt()[0, 0, 0, 0]
            V_t_x = H_funcs.Vt(xt_mod)
            SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
            # print("============================falses.shape = ", falses.shape)
            cond_before_lite = singulars * sigma_next > sigma_0_map_flat
            # print("-----------------------------singulars * sigma_next = ", (singulars * sigma_next).shape)
            # print("============================cond_before_lite.shape = ", cond_before_lite.shape)
            # print("============================singulars.shape = ", singulars.shape)
            # print("===============================cond_before_lite", cond_before_lite)
            cond_after_lite = singulars * sigma_next < sigma_0_map_flat
            # print("===============================cond_after_lite", cond_after_lite)
            cond_before = torch.hstack((cond_before_lite, falses)) # torch.hstack 是 水平拼接
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
            
            diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0_map_flat[cond_before_lite] ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            #missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            #less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = \
                V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0_map_flat)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            
            # print("-------------------------torch.randn_like(U_t_y)[:, cond_before_lite] = ", torch.randn_like(U_t_y)[:, cond_before_lite].shape) # 长度会变
            #noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

            #aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))


    return xs, x0_preds

