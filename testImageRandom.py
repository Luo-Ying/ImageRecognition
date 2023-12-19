import os
from PIL import Image
import tensorflow  as tf
import tensorflow.keras.preprocessing
import numpy as np

def convert_to_jpeg(file_path):
    if file_path.lower().endswith(('.png', '.jpg', '.gif', '.bmp', '.webp')):
        image_path = os.path.join( '', file_path)
    
        img = Image.open(image_path)

        new_path = os.path.splitext(image_path)[0] + ".jpeg"
        img.save(new_path, "JPEG")
        
        img.close()

        os.remove(image_path)
        return new_path
    elif file_path.lower().endswith(('.jpeg')):
        return file_path
        
def load_and_process_from_path_label(image_path):
    print("image path >>>>> ", image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[128,128])
    # image = tf.image.resize(image,[224,224])    # ResNet50, VGG16
    image = tf.image.rgb_to_grayscale(image)      # Décommenter cette ligne pour le model CNN et MLP
    image = image/255.0
    return image

def handle_image_gave(file_path):
    
    image_path = convert_to_jpeg(file_path)
    image = load_and_process_from_path_label(image_path)
    print(image)
    
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
    
    