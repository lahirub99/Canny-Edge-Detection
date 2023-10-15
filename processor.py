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

def convolve(image, kernel):
    """
    Applies a convolution operation to the input image using the given kernel.
    
    Args:
    image: numpy array or list representing the input image
    kernel: numpy array representing the convolution kernel
    
    Returns:
    numpy array representing the convolved image
    """
    
    # Convert the image to a numpy array if it is a list
    if isinstance(image, list):
        image = np.array(image)
    
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the padding needed for the convolution operation
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Create a padded version of the image
    padded_image = np.zeros((image_height + 2*pad_height, image_width + 2*pad_width))
    padded_image[pad_height:-pad_height, pad_width:-pad_width] = image
    
    # Create an output array to hold the convolved image
    output = np.zeros_like(image)
    
    # Apply the convolution operation
    for i in range(image_height):
        for j in range(image_width):
            output[i, j] = np.sum(kernel * padded_image[i:i+kernel_height, j:j+kernel_width])
    
    return output
    
    # # Calculate the padding needed for the convolution operation
    # pad_height = kernel_height // 2
    # pad_width = kernel_width // 2
    
    # # Create a padded version of the image
    # padded_image = np.zeros((image_height + 2*pad_height, image_width + 2*pad_width))
    # padded_image[pad_height:-pad_height, pad_width:-pad_width] = image
    
    # # Create an output array to hold the convolved image
    # output = np.zeros_like(image)
    
    # # Apply the convolution operation
    # for i in range(image_height):
    #     for j in range(image_width):
    #         output[i, j] = np.sum(kernel * padded_image[i:i+kernel_height, j:j+kernel_width])
    
    # return output


def gaussian_kernel(size, sigma=1):
    """
    This function generates a 2D Gaussian kernel of a given size and standard deviation.
    
    Parameters:
    size (int): The size of the kernel (should be an odd integer).
    sigma (float): The standard deviation of the Gaussian distribution (default is 1).
    
    Returns:
    g (numpy.ndarray): A 2D numpy array representing the Gaussian kernel.
    """
    
    # Calculate the center of the kernel
    size = int(size) // 2
    
    # Generate a grid of x and y values
    x, y = np.mgrid[-size:size+1, -size:size+1]
    
    # Calculate the normalizing constant
    normal = 1 / (2.0 * np.pi * sigma**2)
    
    # Calculate the Gaussian distribution
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    
    return g

image_smooth = convolve(image_gray, gaussian_kernel(5, sigma=1.4))
save_image('smooth.jpg', image_smooth, gray=True)


def sobel_filter(img):
    """
    Applies Sobel filter to the input image to detect edges.
    
    Args:
    img: numpy array representing the input image
    
    Returns:
    Tuple containing:
        - G: numpy array representing the gradient magnitude of the image
        - theta: numpy array representing the gradient direction of the image
    """
    
    # Define Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    # Convolve the image with the Sobel kernels
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    
    # Compute gradient magnitude and direction
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

image_gradientCalc, theta = sobel_filter(image_smooth)
save_image('gradientCalc.jpg', image_gradientCalc, gray=True)

def non_max_suppression(img, D):
    """
    Applies non-maximum suppression to the input image using the gradient direction D.

    Args:
    img: numpy array representing the input image
    D: numpy array representing the gradient direction of the input image

    Returns:
    numpy array representing the image after non-maximum suppression
    """
    width, height = img.shape
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
    """
    Applies thresholding to an input image.
    
    Args:
    img (numpy.ndarray): Input image.
    lowThresholdRatio (float): Low threshold ratio. Default is 0.05.
    highThresholdRatio (float): High threshold ratio. Default is 0.09.
    
    Returns:
    Tuple containing:
    - numpy.ndarray: Thresholded image.
    - int: Weak threshold value.
    - int: Strong threshold value.
    """
    
    # Calculate high and low thresholds
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    # Create an empty array to store the result
    width, height = img.shape
    res = np.zeros((width, height), dtype=np.int32)
    
    # Define weak and strong threshold values
    weak = np.int32(25)
    strong = np.int32(255)
    
    # Find indices of pixels above and below the thresholds
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    # Set the threshold values for the corresponding pixels
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

image_threshold, weak, strong = threshold(image_suppression)
save_image('threshold.jpg', image_threshold, gray=True)


def hysteresis(img, weak, strong=255):
    '''
    Transform weak pixels into strong ones only if at least one of the pixels 
    around the one being processed is a strong one, based on the threshold results.

    Args:
    - img: numpy.ndarray - The input image.
    - weak: int - The threshold value for weak pixels.
    - strong: int - The threshold value for strong pixels. Default is 255.

    Returns:
    - numpy.ndarray - The image with weak pixels transformed into strong ones.
    '''

    # Get the width and height of the image
    width, height = img.shape

    # Loop through each pixel in the image
    for i in range(1, width-1):
        for j in range(1, height-1):
            # If the pixel is a weak pixel
            if (img[i,j] == weak):
                try:
                    # Check if any of the surrounding pixels are strong
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        # If at least one surrounding pixel is strong, set the current pixel to strong
                        img[i, j] = strong
                    else:
                        # Otherwise, set the current pixel to 0
                        img[i, j] = 0
                except IndexError as e:
                    # If an IndexError occurs, just pass and continue
                    pass
    # Return the image with weak pixels transformed into strong ones
    return img

image_final = hysteresis(image_threshold, weak, strong)
save_image('final.jpg', image_final, gray=True)
