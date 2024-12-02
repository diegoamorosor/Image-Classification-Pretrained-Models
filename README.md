<p align="center"> 
  <img src="https://i.imgur.com/rs08cGS.png" alt="Image-Classification" /> 
</p>

# Image Classification with Pre-trained Models

Welcome! This repository demonstrates how to classify images using various **pre-trained deep learning models**, such as **InceptionV3**, **ResNet50**, and more. These models are implemented in Python using TensorFlow and Keras, and are designed for tasks like object recognition and image categorization.

---

## 📖 What Are Pre-trained Models?

Pre-trained models are deep learning architectures trained on large datasets, such as **ImageNet**, which contains over 1 million images across 1000 categories. Using pre-trained models allows for:

- **Quick implementation**: No need to train a model from scratch.
- **High accuracy**: Models are optimized for tasks like image recognition.
- **Versatility**: Feature extraction for downstream tasks.

Supported models in this repository include:

| **Model**        | **Input Size** | **Description**                                     |
|-------------------|----------------|-----------------------------------------------------|
| **InceptionV3**   | (299, 299, 3)  | Google’s deep CNN, optimized for image recognition. |
| **ResNet50**      | (224, 224, 3)  | Residual Network, excels in general object recognition. |
| **EfficientNetB0**| (224, 224, 3)  | Balances accuracy and efficiency.                   |
| **MobileNetV2**   | (224, 224, 3)  | Lightweight model for mobile applications.          |
| **VGG16**         | (224, 224, 3)  | Classic architecture for image categorization.      |

---

## 🚀 Features of This Repository

- Load and preprocess images directly from URLs.
- Utilize different pre-trained models with minimal code changes.
- Display the top-5 predictions along with their probabilities.
- Measure and display prediction time for each model.

---

## 📋 Requirements

Ensure you have the following installed:

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Requests
- Pillow

Install dependencies with:

```bash
pip install tensorflow keras numpy requests pillow
````

---

## 📝 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/diegoamorosor/Image-Classification-Pretrained-Models.git
cd Image-Classification-Pretrained-Models
```

### 2. Open the Jupyter Notebook

Run the Jupyter Notebook to follow the step-by-step implementation.

```bash
jupyter notebook
```

### 3. Replace the Image URL

Modify the `image_urls` list in the code to classify images from different links:

```python
image_urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
]
```

### 4. Choose a Model

Modify the `model_name` variable to use a specific model. Supported options include:

- `"InceptionV3"`
- `"ResNet50"`
- `"EfficientNetB0"`
- `"MobileNetV2"`
- `"VGG16"`

```python
model_name = "ResNet50"
```

### 5. Run the Code

Execute the notebook cells to classify the images and see the predictions along with the processing time.

---

## 🖼 Example Output

For the image (https://i.pinimg.com/736x/9b/3a/00/9b3a00eafef25bc831ff13208d54fb56.jpg) using **InceptionV3**, the predictions might look like this:

```text
Processing image from URL: https://i.pinimg.com/736x/9b/3a/00/9b3a00eafef25bc831ff13208d54fb56.jpg
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 231ms/step
Predicciones:
1. pug: 38.44%
2. seat_belt: 25.27%
3. Windsor_tie: 8.53%
4. suit: 5.63%
5. Brabancon_griffon: 1.21%

La imagen fue clasificada como:
pug con una probabilidad de 38.44%

Tiempo de procesamiento: 0.35 segundos
```

---

## 💡 Key Functions

|**Function**|**Description**|
|---|---|
|`load_image_from_url(url)`|Loads an image from a given URL and preprocesses it.|
|`load_model(model_name)`|Loads the specified pre-trained model.|
|`decode_predictions(y)`|Decodes the model's output into readable labels.|

---

## 🔍 Enhancements

Potential improvements include:

1. Adding support for more models.
2. Batch processing for multiple images.
3. Integration with a web interface for user-friendly interaction.
4. Extending to tasks like object detection or segmentation.

---

## 📚 Resources

- [InceptionV3 Paper](https://arxiv.org/abs/1512.00567)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [ImageNet Dataset](http://www.image-net.org/)

---

> ## 🎉 That’s All!
> 
> I hope this guide helps you understand Image Classification better!
