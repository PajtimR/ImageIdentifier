import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

def load_and_preprocess_image(image_path):
    try:
        # Load the image from the file path and resize it to (299, 299
        img = image.load_img(image_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return None

def recognize_image(model, image_paths):
    for image_path in image_paths:
        print(f"\nProcessing image: {image_path}")
        processed_image = load_and_preprocess_image(image_path)

        if processed_image is not None:
            # Make predictions
            predictions = model.predict(processed_image)

            decoded_predictions = decode_predictions(predictions, top=3)[0]
            print("Predictions:")
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions, 1):
                print(f"{i}. {label} ({score:.2f})")

if __name__ == "__main__":
    try:
        # Load the InceptionV3 model pre-trained on ImageNet data
        model = InceptionV3(weights='imagenet')

        # Specify multiple image paths to recognize
        image_paths = [
            "C:/Users/Plus Computers/Desktop/ImageIdentifier(py)/house.jpg",
            "C:/Users/Plus Computers/Desktop/ImageIdentifier(py)/dog.jpg",
            # Add more image paths as needed
        ]

        recognize_image(model, image_paths)

    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
