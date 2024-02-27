import cv2
import numpy as np

image = cv2.imread("LinePhoto.jpg")
print(len(image))
print(image.shape)

print("\n")
print("\n")
print("\n")

# Convert the image to float32 format for numerical calculations
image_float = image.astype(np.float32)

# Brightness
brightness_factor = -100  # Adjust as needed
brightened_image = np.clip(image_float + brightness_factor, 0, 255)

brightened_image = brightened_image.astype(np.uint8)

# Contrast
contrast_factor = 1.5  # Adjust as needed
contrast_image = np.clip(brightened_image * contrast_factor, 0, 255)

contrast_image = contrast_image.astype(np.uint8)

# Blur
blur_factor = 9
blurred_image = cv2.GaussianBlur(contrast_image, (blur_factor, blur_factor), 0)

# Define the threshold range for bright green color
lower_green = np.array([0, 180, 0])  # Lower bound for green in RGB format
upper_green = np.array([190, 255, 190])  # Upper bound for green in RGB format

# Apply thresholding to identify bright green pixels
edImage = cv2.inRange(blurred_image, lower_green, upper_green)


# Convert the data type of the mask to uint8 (0-255 range)
edImage = edImage.astype(np.uint8)

print("Done")
stacked = np.vstack((image,brightened_image,contrast_image))
cv2.imwrite('stacked_image.jpg', stacked)
cv2.imwrite('finished_image.jpg', edImage)



