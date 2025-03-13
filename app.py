from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = tf.keras.models.load_model('C:/fruits/final_mobilenet_fruit_quality_model_finetune.keras')

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] == 4:  # remove alpha channel if exists
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')  # Retrieve the uploaded file from form
        if file:
            img = Image.open(file.stream)  # Open the uploaded image file
            preprocessed_image = preprocess_image(img)  # Preprocess the image for the model
            predictions = model.predict(preprocessed_image)  # Get predictions from the model

            # Assuming your labels are predefined
            class_labels = ['apple', 'banana', 'dragon', 'grapes', 'lemon', 'mango', 'orange', 'papaya', 'pineapple', 'pomegranate', 'strawberry']
            predicted_class_index = np.argmax(predictions)  # Get the predicted class index
            predicted_class = class_labels[predicted_class_index]  # Map index to class label
            ripeness_percentage = np.max(predictions) * 100  # Get the ripeness percentage

            # Example suggestion based on ripeness percentage
            suggestion = f"The {predicted_class} is ripe. Ready to eat!" if ripeness_percentage > 80 else f"The {predicted_class} is not yet ripe."

            # Save the uploaded image to the static/uploads directory
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            img.save(file_path)

            # Return the prediction and image URL as a JSON response
            return jsonify({
                'prediction': predicted_class,
                'ripeness_percentage': ripeness_percentage,
                'suggestion': suggestion,
                'image_url': url_for('static', filename='uploads/' + file.filename)
            })
        else:
            return jsonify({'error': 'No file provided'}), 400
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
