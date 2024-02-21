import numpy as np
from keras.applications.vgg16 import preprocess_input,decode_predictions
from keras.models import load_model

def normalize(input):

    return input / 255.0

def preprocess_image(input_image):
    # Resize the image to 224x224
    image = input_image.resize((224,224))
    # Convert the image to a NumPy array
    array = np.array(image)
    # Expand dimensions
    array = np.expand_dims(array,axis=0)
    # Normalize image array
    array = normalize(array)

    return array

def model_predictions(array):
    # Load trained model
    model = load_model('vgg-16_combined1.h5')
    # Preprocess the input image for the model
    preprocessed_image = preprocess_input(array)
    print(f'Processed Image Shape: {preprocessed_image.shape}')
    # Make predictions
    predictions = model.predict(preprocessed_image)
    # Decode predictions
    #decoded_predictions = decode_predictions(predictions, top=3)[0]

    # for i, (id, label, score) in enumerate(decoded_predictions):
    #     print(f"{i + 1}: {label} ({score:.2f})")

    return predictions