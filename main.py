from flask import Flask, render_template, request
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np # type: ignore
import os

app = Flask(__name__)

# Load your model (Make sure the path is correct)
MODEL_PATH = 'models/cat_vs_dog_models.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    if file:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocess the image: Resize to (224, 224) to match model's expected input size
        img = image.load_img(filepath, target_size=(224, 224))  # Resize to 224x224
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Make the prediction
        prediction = model.predict(img_array)[0][0]
        
        # Determine the result (Dog or Cat)
        result = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

        return render_template('index.html', prediction=result, image_path=filepath)
    else:
        return "No file uploaded", 400

if __name__ == '__main__':
    app.run(debug=True)
