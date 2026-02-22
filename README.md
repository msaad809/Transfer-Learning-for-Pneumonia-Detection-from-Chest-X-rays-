# Transfer Learning for Pneumonia Detection

This project uses ResNet18 to detect pneumonia in chest X-ray images.

- Dataset: Kaggle Chest X-ray Pneumonia URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Preprocessing: Resize, normalize, weighted loss
- Model: ResNet18 with frozen layers and custom classifier
- Evaluation: Confusion matrix, classification report, Grad-CAM
- Environment: Python 3.12, PyTorch 2.x, CUDA, Colab
- Files:
    - Pneumonia_detection_From_Chest_X_rays.ipynb: Main code
    - requirements.txt: Dependencies
    - best_model.pth: Best trained model
    - final_model_checkpoint.pth: Full checkpoint with optimizer state
