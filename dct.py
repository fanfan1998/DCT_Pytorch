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

def rfft_2d(x):
    _, _, height, width = x.size()
    x = torch.fft.rfft2(x, s=(height, width), dim=(2, 3), norm="ortho")
    return x

def irfft_2d(x):
    _, _, height, width = x.size()
    x = torch.fft.irfft2(x, s=(height, width), dim=(2, 3), norm="ortho")
    return x

def split_frequency(x, div_parts):
    _, _, height, width = x.size()

    if div_parts == 0:
        return x
    x = dct_2d(x)
    n, c, w, h = x.size()
    block_size = w // div_parts
    l = []
    for i in range(div_parts):
        for j in range(div_parts):
            temp = idct_2d(x[:, :, i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size])
            l.append(temp)
    x = torch.cat(l, 0)

    return x

def concat_frequency(x, div_parts):
    if div_parts == 0:
        return x
    x = dct_2d(x)
    n, c, w, h = x.size()
    n = n // div_parts ** 2
    r = []
    for i in range(div_parts):
        c = []
        for j in range(div_parts):
            c.append(x[(i * div_parts + j) * n: (i * div_parts + (j + 1)) * n])
        c = torch.cat(c, -1)
        r.append(c)
    x = torch.cat(r, -2)
    x = idct_2d(x)

    return x

if __name__=="__main__":
    # 示例用法
    # 创建一个四维张量
    input_tensor = torch.randn(3, 3, 256, 256)  # 假设形状是(批量大小, 通道数, 高度, 宽度)

    # 对输入张量进行DCT变换
    tensor1 = rfft_2d(input_tensor)
    tensor2 = irfft_2d(tensor1)
    print(tensor2.shape)
    print(torch.allclose(input_tensor, tensor2, atol=1e-6))
    #
    # tensor3 = split_frequency(input_tensor,2)
    # print(tensor3.shape)
    # tensor4 = concat_frequency(tensor3,2)
    # print(tensor4.shape)
    # print(torch.allclose(input_tensor, tensor4, atol=1e-6))
