#----------------------------------------------------------
# Name: Ganegoda G.S.S.S
# RegNo: EG/2019/3588
# Take Home Assignment 2
#----------------------------------------------------------

# Import required libraries
import cv2
import numpy as np

def region_growing_with_error(image, seed_points, error_threshold, patch_size):
    # Initialize the segmented image with zeros
    segmented_image = np.zeros_like(image)
    visited = np.zeros_like(image, dtype=bool)

    # Create a list to store region means
    region_means = [image[seed_point[1], seed_point[0]] for seed_point in seed_points]

    # Define a function to calculate the mean of a region
    def calculate_region_mean(region):
        return np.mean(region)

    # Define a function to check if a pixel is within the specified error range of the region mean
    def within_error_range(pixel_value, region_mean):
        return abs(pixel_value - region_mean) <= error_threshold

    # Define a function to check if a pixel is within the image boundaries
    def within_image_boundaries(x, y):
        return 0 <= x < image.shape[1] and 0 <= y < image.shape[0]

    # Define a function to get neighboring pixels of a given pixel within a patch size
    def get_neighboring_pixels(x, y, patch_size):
        half_patch = patch_size // 2
        neighbors = []
        for i in range(-half_patch, half_patch + 1):
            for j in range(-half_patch, half_patch + 1):
                if i == 0 and j == 0:
                    continue
                if within_image_boundaries(x + i, y + j):
                    neighbors.append((x + i, y + j))
        return neighbors

    # Define a function to grow a region from a seed point
    def grow_region(seed_point_idx):
        seed_point = seed_points[seed_point_idx]
        region = [seed_point]
        region_mean = region_means[seed_point_idx]

        while region:
            current_pixel = region.pop(0)
            x, y = current_pixel

            # Check if the pixel has been visited
            if not visited[y, x]:
                visited[y, x] = True

                # Check neighboring pixels within the patch size
                for neighbor in get_neighboring_pixels(x, y, patch_size):
                    nx, ny = neighbor
                    if not visited[ny, nx]:
                        pixel_value = image[ny, nx]
                        if within_error_range(pixel_value, region_mean):
                            region.append(neighbor)
                            segmented_image[ny, nx] = 255
                            visited[ny, nx] = True
                            # Update region mean
                            region_mean = calculate_region_mean([image[p[1], p[0]] for p in region])
                # Update region mean
                region_means[seed_point_idx] = region_mean

    # Grow regions from each seed point
    for i, seed_point in enumerate(seed_points):
        grow_region(i)

    return segmented_image

image = cv2.imread('data/sample.jpg', 0)
ret, img = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('Input', img)
cv2.waitKey()

seed_points = [(125, 100)]  # seed points
error_threshold = 10  # error threshold
patch_size = 64  # patch size

segmented_image = region_growing_with_error(image, seed_points, error_threshold, patch_size)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
