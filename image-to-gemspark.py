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
    
def image_to_gemspark_HSV(image_path, output_path, size_multiplier=None, height=None, width=None, resize=None):
    # Load the images
    image = cv2.imread(image_path) # load image from path

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to load.")

    if size_multiplier:
        image = cv2.resize(image, (int(image.shape[1]*size_multiplier), int(image.shape[0]*size_multiplier))) # resize by multiplier if specified
    elif height and width:
        image = cv2.resize(image, (width, height))

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32) 

    # Normalize S and V to 0-1 range for consistent weighting
    h_rad = img_hsv[:, :, 0] * (2 * np.pi / 180.0) # Convert Hue to Radians (0 to 2pi)
    s = img_hsv[:, :, 1] / 255.0
    v = img_hsv[:, :, 2] / 255.0

    # This fixes the circular Hue problem
    # S_FACTOR: Controls how wide the cylinder is.
    # Higher number = Saturation is MORE important.
    S_FACTOR = 1.7

    img_x = (s * S_FACTOR) * np.cos(h_rad)
    img_y = (s * S_FACTOR) * np.sin(h_rad)

    # V_FACTOR: Controls how tall the cylinder is.
    # Higher number = Brightness is MORE important.
    V_FACTOR = 0.7

    img_z = v * V_FACTOR

    img_projected = np.dstack((img_x, img_y, img_z))

    palette = np.array(get_gemspark_colors(), dtype=np.uint8).reshape(1, -1, 3)
    palette_hsv = cv2.cvtColor(palette, cv2.COLOR_BGR2HSV).astype(np.float32)

    p_h_rad = palette_hsv[:, :, 0] * (2 * np.pi / 180.0)
    p_s = palette_hsv[:, :, 1] / 255.0
    p_v = palette_hsv[:, :, 2] / 255.0
    
    p_x = p_s * np.cos(p_h_rad)
    p_y = p_s * np.sin(p_h_rad)
    p_z = p_v

    palette_projected = np.dstack((p_x, p_y, p_z)).reshape(1, 1, -1, 3)

    diff = img_projected[:, :, np.newaxis, :] - palette_projected
    dist_sq = np.sum(diff**2, axis=-1)  ** 0.5

    indices = np.argmin(dist_sq, axis=2)
    palette_bgr_flat = np.array(get_gemspark_colors(), dtype=np.uint8)
    res = palette_bgr_flat[indices].astype(np.uint8)

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

image_to_gemspark_HSV('stuff/image.png', 'output.png', size_multiplier=0.5)