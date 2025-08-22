import numpy as np

loaded = np.load("inp_masks/lolcat_extra.npy")

print("--------------------------------------loaded.shape = ", loaded.shape)
# 如果是二维或三维 mask，先reshape成二维，每行一个元素序列
if loaded.ndim > 2:
    reshaped = loaded.reshape(loaded.shape[0], -1)  # 每张 mask 展平成一行
else:
    reshaped = loaded

# 保存到 txt 文件
np.savetxt("mask_output.txt", reshaped, fmt='%d')  # fmt='%d' 如果是整数 mask
