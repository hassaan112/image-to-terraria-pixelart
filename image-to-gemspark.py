import cv2
import os

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

def image_to_gemspark(image_path, output_path, size=1):
    # Load the images
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to load.")
    



# image_to_gemspark('TerrariaColors.png', 'output.png')