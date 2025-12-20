import cv2
import os
import numpy as np
from random import sample

def get_unique_colors(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    imgList = image.tolist()[0]
    colors = []
    for color in imgList:
        if color not in colors:
            colors.append(color)
    return colors

def get_gemspark_colors():
    return get_unique_colors(os.path.join('references','TerrariaGemsparkColors.png'))

def image_to_gemspark(image_path, output_path, size_multiplier=None, height=None, width=None, resize=None):
    # Load the images
    image = cv2.imread(image_path) # load image from path

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to load.")

    if size_multiplier:
        image = cv2.resize(image, (int(image.shape[1]*size_multiplier), int(image.shape[0]*size_multiplier))) # resize by multiplier if specified
    elif height and width:
        image = cv2.resize(image, (width, height))

    data = image.astype(np.float32) 
    palette = np.array(get_gemspark_colors(), np.float32)

    diff = data[:, :, np.newaxis, :] - palette
    dist_sq = np.sum(diff**2, axis=-1)

    indices = np.argmin(dist_sq, axis=2)
    res = palette[indices].astype(np.uint8)

    if resize == True:
        height, width = res.shape[:2]
        factor = 1080 / height 
        print(width, height, factor)
        if height > 1080:
            res = cv2.resize(res, (width*factor, height*factor), interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(res, (width*factor, height*factor), interpolation=cv2.INTER_CUBIC)
        

    cv2.imshow("Preview", res)
    cv2.waitKey(0)
    cv2.imwrite(output_path, res)

def image_to_gemspark_LAB(image_path, output_path, size_multiplier=None, height=None, width=None, resize=None):
    # Load the images
    image = cv2.imread(image_path) # load image from path

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to load.")

    if size_multiplier:
        image = cv2.resize(image, (int(image.shape[1]*size_multiplier), int(image.shape[0]*size_multiplier))) # resize by multiplier if specified
    elif height and width:
        image = cv2.resize(image, (width, height))

    img = image.astype(np.float32) / 255.0
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    palette = np.array(get_gemspark_colors(), np.float32) / 255.0
    palette_as_img = palette[np.newaxis, :, :] # to avoid some weird errors
    palette_lab = cv2.cvtColor(palette_as_img, cv2.COLOR_BGR2LAB)
    palette_lab = palette_lab.reshape(-1, 3)

    diff = img_lab[:, :, np.newaxis, :] - palette_lab
    dist_sq = np.sum(diff**2, axis=-1)

    indices = np.argmin(dist_sq, axis=2)
    res = (palette[indices] * 255).astype(np.uint8)

    if resize == True:
        height, width = res.shape[:2]
        factor = 1080 / height 
        print(width, height, factor)
        if height > 1080:
            res = cv2.resize(res, (width*factor, height*factor), interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(res, (width*factor, height*factor), interpolation=cv2.INTER_CUBIC)
        

    cv2.imshow("Preview", res)
    cv2.waitKey(0)
    cv2.imwrite(output_path, res)
    
# def closest_color_euclidean(input_color, color_list):
#     min_distance = float('inf')
#     closest_color = None
#     for color in color_list:
#         distance = sum((ic - cc) ** 2 for ic, cc in zip(input_color, color)) ** 0.5
#         if distance < min_distance:
#             min_distance = distance
#             closest_color = color
#     return closest_color

# def closest_color_euclidean_weighted(input_color, color_list):
#     min_distance = float('inf')
#     closest_color = None
#     weightR, weightG, weightB = 2, 4, 3
#     for color in color_list:
#         distance = (weightR*((input_color[0]-color[0])**2) + weightG*((input_color[1]-color[1])**2) + weightB*((input_color[2]-color[2])**2)) ** 0.5
#         if distance < min_distance:
#             min_distance = distance
#             closest_color = color
#     return closest_color

image_to_gemspark('stuff/ado.jpg', 'output.png', size_multiplier=0.125)