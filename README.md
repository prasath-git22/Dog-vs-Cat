# 🐶🐱 Dog vs Cat Image Classification using CNN

A deep learning-based image classification project that uses a **Convolutional Neural Network (CNN)** to classify images as either a **Dog 🐶** or a **Cat 🐱**.

The project includes a trained TensorFlow/Keras model and a Flask web application where users can upload an image and receive a prediction.

---

## 📌 Project Overview

Image classification is one of the important applications of Computer Vision and Deep Learning.

In this project, a CNN model is trained to learn visual patterns from dog and cat images and classify an unseen image into one of two classes:

- 🐶 Dog
- 🐱 Cat

A Flask-based web interface is provided so users can upload an image and get the model's prediction.

---

## ✨ Features

- 🖼️ Upload an image through a web interface
- 🐶 Classify images as Dog
- 🐱 Classify images as Cat
- 🧠 CNN-based deep learning model
- 🔄 Automatic image preprocessing
- 📐 Resize input images to 224 × 224
- 📊 Normalize image pixel values
- 🌐 Flask web application
- ⚡ Real-time prediction

---

## 🧠 Model Workflow

```text
Input Image
     │
     ▼
Image Upload
     │
     ▼
Resize to 224 × 224
     │
     ▼
Convert Image to Array
     │
     ▼
Normalize Pixel Values
     │
     ▼
CNN Model
     │
     ▼
Prediction Probability
     │
     ├───────────────┐
     ▼               ▼
   Cat 🐱          Dog 🐶
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| NumPy | Numerical operations |
| Flask | Web application |
| HTML/CSS | User interface |
| OpenCV / Image Processing | Image preprocessing |

---

## 📂 Project Structure

```text
Dog-vs-Cat/
│
├── models/
│   └── cat_vs_dog_models.keras
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── main.py
│
└── README.md
```

---

## ⚙️ How It Works

### 1. Upload Image

The user uploads a dog or cat image through the Flask web application.

### 2. Image Preprocessing

The uploaded image is:

- Loaded using Keras
- Resized to `224 × 224`
- Converted into a NumPy array
- Normalized by dividing pixel values by `255`

### 3. CNN Prediction

The processed image is passed to the trained CNN model.

The model generates a prediction probability.

### 4. Classification

The prediction is classified using a threshold:

```python
prediction > 0.5
```

If the probability is greater than `0.5`:

```text
Dog 🐶
```

Otherwise:

```text
Cat 🐱
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/prasath-git22/Dog-vs-Cat.git
```

```bash
cd Dog-vs-Cat
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the required libraries:

```bash
pip install tensorflow flask numpy pillow
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python main.py
```

Then open:

```text
http://127.0.0.1:5000
```

Upload an image and the application will display the predicted class.

---

## 📸 Prediction

Example:

```text
Input:
[Uploaded Dog Image]

Prediction:
🐶 Dog
```

or

```text
Input:
[Uploaded Cat Image]

Prediction:
🐱 Cat
```

---

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Deep Learning
- Convolutional Neural Networks
- Image Classification
- TensorFlow and Keras
- Image preprocessing
- NumPy
- Model prediction
- Flask deployment
- Building an end-to-end AI application

---

## 🔮 Future Improvements

Possible improvements include:

- [ ] Add prediction confidence percentage
- [ ] Display prediction probability
- [ ] Add model accuracy and loss graphs
- [ ] Improve UI design
- [ ] Add drag-and-drop image upload
- [ ] Deploy the application online
- [ ] Add more animal classes
- [ ] Improve the model using data augmentation
- [ ] Experiment with transfer learning

---

## 👨‍💻 Author

**Prasath**

B.Tech – Artificial Intelligence and Data Science

GitHub:

https://github.com/prasath-git22

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ star.

---

## 📜 License

This project is created for educational and portfolio purposes.
