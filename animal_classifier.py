import pathlib
import numpy as np
import tensorflow as tf

folder = "C:\\Users\\user\\Desktop\\split_dataset"

dataset_dir = pathlib.Path(folder)

batch_size = 32
img_width = 224
img_height = 224

train = tf.keras.utils.image_dataset_from_directory(
    dataset_dir / 'train',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

class_names_len = len(train.class_names)
class_names = train.class_names


class AnimalClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model('Models/animal_model__.h5')
        self.class_names = class_names

    def preprocess_image(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def predict(self, image_path):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        top_class = self.class_names[predicted_class]
        top_probability = round(predictions[0][predicted_class] * 100, 2)
        return top_class, top_probability