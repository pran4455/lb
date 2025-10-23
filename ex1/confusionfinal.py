from pycm import ConfusionMatrix

# Random confusion matrix for Brain Tumor MRI dataset
confusion_matrix = {
    "Glioma": {"Glioma": 120, "Meningioma": 5, "No Tumor": 3, "Pituitary": 7},
    "Meningioma": {"Glioma": 4, "Meningioma": 115, "No Tumor": 6, "Pituitary": 5},
    "No Tumor": {"Glioma": 2, "Meningioma": 4, "No Tumor": 130, "Pituitary": 4},
    "Pituitary": {"Glioma": 5, "Meningioma": 6, "No Tumor": 5, "Pituitary": 120}
}

# Create confusion matrix object
cm = ConfusionMatrix(matrix=confusion_matrix)

# Display metrics
print(cm)
