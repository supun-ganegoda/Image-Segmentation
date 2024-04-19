#----------------------------------------------------------
# Name: Ganegoda G.S.S.S
# RegNo: EG/2019/3588
# Take Home Assignment 2
#----------------------------------------------------------

# Import required libraries
import numpy as np
import cv2

def generateImage(width, height):
    # Create an empty grayscale image
    image = np.zeros((height, width), dtype=np.uint8)

    # Assign pixel values for different regions
    # Background (Black)
    image[:, :] = 255

    # Square (Gray)
    square_size = width // 2
    square_x = 0 
    square_y = (height - square_size) // 2
    image[square_y:square_y+square_size, square_x:square_x+square_size] = 128  # Gray color

    # Circle (White)
    circle_radius = width // 5
    circle_center = (width - circle_radius, height // 2)
    cv2.circle(image, circle_center, circle_radius, 0, -1)  # white color

    return image

def addGaussianNoise(image):
    # Define mean and standard deviation
    mean = 0
    stddev = 50
    # Convert image to floating-point data type
    image_float = image.astype(np.float32)
    # Generate Gaussian noise with specified mean and standard deviation
    noise = np.random.normal(mean, stddev, size=image.shape).astype(np.float32)
    # Add noise to the image
    img_noised = image_float + noise
    # Clip pixel values to ensure they are within the valid range [0, 255]
    noisy_image = np.clip(img_noised, 0, 255).astype(np.uint8)
    return noisy_image

# Display the image
generatedImage = generateImage(300,300)
cv2.imshow("Image with 3 Pixel Values", generatedImage)

# Display the noisy image
noisyImage = addGaussianNoise(generatedImage)
cv2.imshow("Noise added Image", noisyImage)

# Apply Otsu's thresholding
_, otsuThreshold = cv2.threshold(noisyImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the thresholded image
cv2.imshow("Otsu's Thresholding", otsuThreshold)

cv2.waitKey(0)
cv2.destroyAllWindows()

