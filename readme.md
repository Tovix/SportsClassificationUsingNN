# Sports Classification Using CNNs

## Project Summary

This project explores the use of Convolutional Neural Networks (CNNs) for classifying sports images. Multiple architectures were evaluated to determine the most effective approach for this task.

---

## Dataset & Preprocessing

- **Image Resizing:** All images were resized to match model input requirements using Keras’ `ImageDataGenerator`.
- **Format Unification:** All images were converted to PNG to ensure consistency and avoid format conflicts.
- **Data Augmentation:**  
  - Techniques applied: rotation, shear, zoom, width/height shift, horizontal flip, and fill mode.
  - The dataset was expanded to approximately **22,000 images** through augmentation.  
  - **Note:** Accuracy plateaued beyond this dataset size, indicating diminishing returns from further augmentation.

---

## Models Evaluated & Results

### 1. Basic CNN

- **Parameters:** ~2.1M
- **Accuracy:**
  - First 5 epochs: **62%** (train), **74%** (validation)
  - Last 5 epochs: **61%** (train), **70%** (validation)
- **Test Set (1,327 images):**  
  - **75%** accuracy, **64%** validation accuracy
- **Remarks:**  
  - Best suited for small datasets due to lower complexity and faster training.

---

### 2. InceptionV3

- **Model Depth:** Significantly deeper with higher complexity.
- **Accuracy:**
  - First 5 epochs: **77%** (train), **89%** (validation)
  - Next 5 epochs: **86%** (train), **92%** (validation)
  - After extended training (5x4 epochs): **95%** accuracy, **84%** test accuracy
- **Remarks:**  
  - Best-performing model across all scenarios.

---

### 3. VGG16

- **Accuracy after 5 epochs:** **19%** (train), **50%** (validation)
- **Remarks:**  
  - Underperformed in this setup; not well-suited for this dataset or configuration.

---

## Final Conclusion

- **InceptionV3** was the most effective model, demonstrating robust performance on both small and large datasets, and achieving the highest test accuracy (**84%**).
- **Basic CNN** is acceptable for lightweight, small-data use cases.
- **VGG16** underperformed and was not a good fit for this experiment.

---

## Key Takeaways

- **Model selection** and **data preprocessing** are critical for optimal performance in image classification tasks.
- **InceptionV3** is recommended for similar sports image classification problems, especially when dataset size and computational resources allow.

---

## How to Run

1. Clone this repository.
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Prepare your dataset in the specified format.
4. Run the training script for your chosen model.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.