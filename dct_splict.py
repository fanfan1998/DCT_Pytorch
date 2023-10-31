import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt
def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m

def dct_2d(x):
    height, width = x.size()
    if x.is_cuda:
        dct = torch.tensor(DCT_mat(height)).float().cuda()
        dctt = torch.transpose(torch.tensor(DCT_mat(width)).float(), 0, 1).cuda()
    else:
        dct = torch.tensor(DCT_mat(height)).float()
        dctt = torch.transpose(torch.tensor(DCT_mat(width)).float(), 0, 1)
    return dct @ x @ dctt

def idct_2d(x):
    height, width = x.size()
    if x.is_cuda:
        dct = torch.tensor(DCT_mat(width)).float().cuda()
        dctt = torch.transpose(torch.tensor(DCT_mat(height)).float(), 0, 1).cuda()
    else:
        dct = torch.tensor(DCT_mat(width)).float()
        dctt = torch.transpose(torch.tensor(DCT_mat(height)).float(), 0, 1)
    return dctt @ x @ dct


def split_dct(x, div_parts):
    if div_parts == 0:
        return x
    w, h = x.size()
    block_size = w // div_parts
    l = []
    for i in range(div_parts):
        for j in range(div_parts):
            temp = x[ i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]
            l.append(temp)

    return l


# Read the image in color (RGB)
image_path = r"C:\Users\84544\Desktop\000.png"  # Replace with your image path
image = Image.open(image_path)

# Separate the image into RGB channels
r, g, b = image.split()

# Get the size of the original image
original_size = image.size

# Apply DCT to each channel
r_dct = dct_2d(torch.tensor(np.array(r)).float())
g_dct = dct_2d(torch.tensor(np.array(g)).float())
b_dct = dct_2d(torch.tensor(np.array(b)).float())

# Split each DCT coefficient into n frequency bands
n = 2  # Number of frequency bands
r_frequency_bands = split_dct(r_dct, n)
g_frequency_bands = split_dct(g_dct, n)
b_frequency_bands = split_dct(b_dct, n)

# Create a directory to save the output images
output_dir = "output_frequency_bands"
os.makedirs(output_dir, exist_ok=True)

# Save each frequency band as an image
l_images = []  # List to store individual frequency band images

for i, (r_band, g_band, b_band) in enumerate(zip(r_frequency_bands, g_frequency_bands, b_frequency_bands)):
    # Apply IDCT to each channel
    r_idct = idct_2d(r_band)
    g_idct = idct_2d(g_band)
    b_idct = idct_2d(b_band)

    # Merge the channels
    reconstructed_image = Image.merge("RGB", (
        Image.fromarray(r_idct.numpy().astype('uint8')),
        Image.fromarray(g_idct.numpy().astype('uint8')),
        Image.fromarray(b_idct.numpy().astype('uint8'))
    ))

    # reconstructed_image = Image.fromarray((r_idct.numpy().astype('uint8') + g_idct.numpy().astype('uint8') + b_idct.numpy().astype('uint8')) / 3.0)

    # Resize the reconstructed image to the original size
    reconstructed_image = reconstructed_image

    # Append the reconstructed image to the list
    l_images.append(reconstructed_image)

x_offset, y_offset = 0, 0
# Create a montage of all frequency band images
montage = Image.new("RGB", original_size)
for img in l_images:
    montage.paste(img, (x_offset, y_offset))
    x_offset += img.width
    if x_offset >= montage.width:
        x_offset = 0
        y_offset += img.height

# Save the montage as a single image
montage_path = os.path.join(output_dir, "frequency_bands_montage.png")
montage.save(montage_path)

# Display the montage using Matplotlib
montage_img = Image.open(montage_path)
plt.imshow(montage_img)
plt.axis('off')  # Hide the axis
plt.show()
