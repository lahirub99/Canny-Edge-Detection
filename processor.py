from scipy import ndimage
from scipy.ndimage import convolve

import numpy as np
import matplotlib.pyplot as plt

''' Saving a image in the disk '''
def save_image(filename, image, gray=False):
    temp = np.array(image)
    temp = temp.astype(np.uint8)
    # Save the image
    path = image_path + filename   # separate folder created for output images
    if gray:    
        plt.imsave(path, temp, cmap='gray')
    else:       
        plt.imsave(path, temp)
    print(f"Image saved at {path}")
    return

# Reading in images
image_path = 'images/'

# Load the original image
image_rgb = plt.imread('images/original.jpg')
width, height, channels = image_rgb.shape

#print(width, height, channels)



# Convert to grayscale
def graysclale():
    image_gray = [[[0 for k in range(3)] 
                    for j in range(height)] 
                    for i in range(width)]

    for i in range(width):
        for j in range(height):
            # Extracting the RGB values
            r, g, b = image_rgb[i][j]
            # Converting to grayscale considering the Luminance level as it was widely use than others in as in YUV and YCrCb formats
            # Formula: Y = 0.299 R + 0.587 G + 0.114 B
            image_gray[i][j] = int(0.299*r + 0.587*g + 0.114*b)

            ### Another methods to convert to grayscale:
            ## 1. Average method
            # image_rgb[i][j] = int(sum(r + g + b)/3)
            ## 2. Lightness method
            # image_rgb[i][j] = int((max(r, g, b) + min(r, g, b))/2)
    else:
        save_image('grayscale.jpg', image_gray, gray=True)
        print('Image converted to grayscale successfully!')
        return image_gray

image_gray = graysclale()


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

image_smooth = convolve(image_gray, gaussian_kernel(5, sigma=1.4))
save_image('smooth.jpg', image_smooth, gray=True)


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

image_gradientCalc, theta = sobel_filters(image_smooth)
save_image('gradientCalc.jpg', image_gradientCalc, gray=True)


def non_max_suppression(img, D):
    Z = np.zeros((width, height), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1,width-1):
        for j in range(1,height-1): 
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

image_suppression = non_max_suppression(image_gradientCalc, theta)
save_image('suppression.jpg', image_suppression, gray=True)

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    res = np.zeros((width, height), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

image_threshold, weak, strong = threshold(image_suppression)
save_image('threshold.jpg', image_threshold, gray=True)

'''Transform weak pixels into strong ones only if at least one of the pixels around the one being processed is a strong one, based on the threshold results.'''
def hysteresis(img, weak, strong=255):
    for i in range(1, width-1):
        for j in range(1, height-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

image_final = hysteresis(image_threshold, weak, strong)
save_image('final.jpg', image_final, gray=True)
