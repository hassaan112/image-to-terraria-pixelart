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
    
def closest_color_euclidean(input_color, color_list):
        min_distance = float('inf')
        closest_color = None
        for color in color_list:
            distance = sum((ic - cc) ** 2 for ic, cc in zip(input_color, color)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        return closest_color


# image_to_gemspark('TerrariaColors.png', 'output.png')