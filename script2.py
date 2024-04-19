#----------------------------------------------------------
# Name: Ganegoda G.S.S.S
# RegNo: EG/2019/3588
# Take Home Assignment 2
#----------------------------------------------------------

# Import required libraries
import cv2
import numpy as np

def show_segmentation(mask):
    cv2.imshow('Segmentation Process', mask)
    cv2.waitKey(1)  # Keep the window open

def region_growing(image, seed_points, threshold_range):
    # Create a mask to store the segmented region
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Create a queue to store the seed points
    queue = []
    for seed_point in seed_points:
        queue.append(seed_point)
    
    iteration = 0
    # Perform the region growing
    while queue:
        # Increment iteration count
        iteration += 1
        # Pop a seed point from the queue
        current_point = queue.pop(0)
        
        # Get the pixel value at the current point
        current_value = image[current_point[1], current_point[0]]
        
        # Add the current point to the mask
        mask[current_point[1], current_point[0]] = 255

        # Display the current state of the segmentation mask
        if iteration % 10 == 0:  # Update every 10 iterations
            show_segmentation(mask)
        
        # Check neighbors of the current point
        for i in range(-1, 2):
            for j in range(-1, 2):
                # Skip the current pixel
                if i == 0 and j == 0:
                    continue
                
                # Get the coordinates of the neighbor
                x = current_point[0] + i
                y = current_point[1] + j
                
                # Check if the neighbor is within the image boundaries
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    # Check if the neighbor is within the threshold range
                    neighbor_value = image[y, x]
                    if np.abs(neighbor_value - current_value) <= threshold_range:
                        # Check if the neighbor is not visited yet
                        if mask[y, x] == 0:
                            # Add the neighbor to the queue
                            queue.append((x, y))
                            # Mark the neighbor as visited
                            mask[y, x] = 255
    
    return mask

# Load the image
image = cv2.imread('results/Q2/input.jpg', cv2.IMREAD_GRAYSCALE)

# Define seed points 
seed_points = [(490, 200), (680,170), (400,400)]  

# Define threshold range for pixel values
threshold_range = 10

# Perform region-growing
segmented_image = region_growing(image, seed_points, threshold_range)

cv2.destroyAllWindows()
# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
