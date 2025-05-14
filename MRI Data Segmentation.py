import numpy as np 
import matplotlib.pyplot as plt 
from skimage import filters, morphology, measure, draw 
from skimage.segmentation import clear_border 
from skimage.util import random_noise 
from skimage.color import label2rgb
 # Step 1: Generate a synthetic "MRI-like" image 
def generate_synthetic_mri(size=256):
    image = np.zeros((size, size), dtype=np.float32)
    # Add a circular "brain" region 
    rr, cc = draw.disk((size // 2, size // 2), size // 3) 
    image[rr, cc] = 0.6
    # Add synthetic "tumor" region 
    rr, cc = draw.disk((size // 2 + 30, size // 2), size // 10) 
    image[rr, cc] = 0.9 
    # Add Gaussian noise
    image = random_noise(image, mode='gaussian', var=0.01)
    return image
    # Step 2: Segment using Otsu's thresholding
def segment_mri(image): 
    threshold = filters.threshold_otsu(image) 
    binary = image > threshold 
    binary = morphology.remove_small_objects(binary, min_size=500) 
    binary = morphology.remove_small_holes(binary, area_threshold=300) 
    binary = clear_border(binary) 
    return binary
    # Step 3: Label and display results 
def display_segmentation(image, mask):
    labeled_mask = measure.label(mask) 
    image_overlay = label2rgb(labeled_mask, image=image, bg_label=0)
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image, cmap='gray') 
    ax[0].set_title('Synthetic MRI Image')
    ax[1].imshow(mask, cmap='gray') 
    ax[1].set_title('Binary Mask')
    ax[2].imshow(image_overlay) 
    ax[2].set_title('Segmented Overlay') 
    
    for a in ax:
        a.axis('off') 
    plt.tight_layout() 
    plt.show() 
    
    # Run the script 
if __name__ == "__main__": 
    mri_image = generate_synthetic_mri()
    segmentation_mask = segment_mri(mri_image) 
    display_segmentation(mri_image, segmentation_mask)
