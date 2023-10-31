import torch
import numpy as np

def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m

def dct_2d(x):
    _, _, height, width = x.size()
    if x.is_cuda:
        dct = torch.tensor(DCT_mat(height)).float().cuda()
        dctt = torch.transpose(torch.tensor(DCT_mat(width)).float(), 0, 1).cuda()
    else:
        dct = torch.tensor(DCT_mat(height)).float()
        dctt = torch.transpose(torch.tensor(DCT_mat(width)).float(), 0, 1)
    return dct @ x @ dctt

def idct_2d(x):
    _, _, height, width = x.size()
    if x.is_cuda:
        dct = torch.tensor(DCT_mat(width)).float().cuda()
        dctt = torch.transpose(torch.tensor(DCT_mat(height)).float(), 0, 1).cuda()
    else:
        dct = torch.tensor(DCT_mat(width)).float()
        dctt = torch.transpose(torch.tensor(DCT_mat(height)).float(), 0, 1)
    return dctt @ x @ dct


if __name__=="__main__":
    # 示例用法
    # 创建一个四维张量
    input_tensor = torch.randn(3, 3, 256, 256)  # 假设形状是(批量大小, 通道数, 高度, 宽度)

    # 对输入张量进行DCT变换
    tensor1 = dct_2d(input_tensor)
    tensor2 = idct_2d(tensor1)
    print(torch.allclose(input_tensor, tensor2, atol=1e-6))
