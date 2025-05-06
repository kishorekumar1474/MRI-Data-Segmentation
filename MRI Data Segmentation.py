# Step 1: Install required libraries
!pip install SimpleITK matplotlib

# Step 2: Upload the MRI image file (.nii or .nii.gz)
from google.colab import files
uploaded = files.upload()

# Step 3: Read the uploaded file
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Get the uploaded filename (assumes only one file uploaded)
filename = next(iter(uploaded))

# Load the MRI image
image = sitk.ReadImage(filename)
image = sitk.Cast(image, sitk.sitkFloat32)  # Ensure it's in float32 format

# Step 4: Apply Otsu threshold segmentation
otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)
otsu_filter.SetOutsideValue(1)
segmented = otsu_filter.Execute(image)

# Step 5: Save the segmented output
output_filename = "segmented_output.nii.gz"
sitk.WriteImage(segmented, output_filename)
print(f"Segmented image saved as: {output_filename}")

# Step 6: Visualize a middle slice
image_array = sitk.GetArrayFromImage(image)
segmented_array = sitk.GetArrayFromImage(segmented)

# Display middle slice
slice_index = image_array.shape[0] // 2

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original MRI Slice")
plt.imshow(image_array[slice_index], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmented Slice (Otsu)")
plt.imshow(segmented_array[slice_index], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
