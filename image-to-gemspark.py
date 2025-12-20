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

def image_to_gemspark(image_path, output_path, size=1):
    # Load the images
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = cv2.cvtCol
    image = cv2.resize(image, (image.shape[1]//size, image.shape[0]//size))

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    image = image.tolist()
    print(len(image), len(image[0]))
    for row in range(len(image)):
        if row % 10 == 0:
            print(row)
        for col in range(len(image[0])):
            original_color = image[row][col] #rgb color
            new_color = closest_color_euclidean(original_color, get_gemspark_colors())
            image[row][col] = new_color  
    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("ae", image)
    cv2.waitKey(0)
    
    # print("Available Colors: ", get_gemspark_colors())
    # cols = sample(image.tolist()[0], 1000)
    # for color in cols:
    #     euclidean = closest_color_euclidean(color, get_gemspark_colors())
    #     weighted = closest_color_euclidean_weighted(color, get_gemspark_colors())
    #     if euclidean != weighted:
    #         print("Original Color:", color)
    #         print("Closest Color:", closest_color_euclidean(color, get_gemspark_colors()))
    #         print("Closest Color (Weighted):", closest_color_euclidean_weighted(color, get_gemspark_colors()))
    
def closest_color_euclidean(input_color, color_list):
    min_distance = float('inf')
    closest_color = None
    for color in color_list:
        distance = sum((ic - cc) ** 2 for ic, cc in zip(input_color, color)) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

def closest_color_euclidean_weighted(input_color, color_list):
    min_distance = float('inf')
    closest_color = None
    weightR, weightG, weightB = 2, 4, 3
    for color in color_list:
        distance = (weightR*((input_color[0]-color[0])**2) + weightG*((input_color[1]-color[1])**2) + weightB*((input_color[2]-color[2])**2)) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

image_to_gemspark('references/nazuna2.png', 'output.png')