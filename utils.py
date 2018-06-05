import cv2

import torch
import torchvision.transforms as transforms

from model_basics import binarize_mask, get_mask

def get_binarized_mask(input, model):
    mask = get_mask(input, model)
    return binarize_mask(mask.clone())

def get_masked_images(input, binary_mask, gray_scale = 0):
    with torch.no_grad():
        if gray_scale > 0:
            gray_background = torch.zeros_like(input) + 0.35
            masked_in = binary_mask*input + (1-binary_mask)*gray_background
            masked_out = (1-binary_mask)*input + binary_mask*gray_background
        else:
            masked_in = binary_mask*input
            masked_out = (1-binary_mask)*input
        
        return masked_in, masked_out

def inpaint(mask, masked_image):
    l = []
    for i in range(mask.size(0)):
        permuted_image = permute_image(masked_image[i], mul255=True)
        m = mask[i].squeeze().byte().numpy()
        inpainted_numpy = cv2.inpaint(permuted_image, m, 3, cv2.INPAINT_TELEA) #cv2.INPAINT_NS
        l.append(transforms.ToTensor()(inpainted_numpy).unsqueeze(0))
    inpainted_tensor = torch.cat(l, 0)
    return inpainted_tensor       

def permute_image(image_tensor, mul255 = False):
    with torch.no_grad():
        image = image_tensor.clone().squeeze().permute(1, 2, 0)
        if mul255:
            image *= 255
            image = image.byte()
        return image.numpy()
