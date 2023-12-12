from PIL import Image
import tensorflow  as tf
import os
import json
import pathlib
import random

def convert_to_jpeg(image_path):
    img = Image.open(image_path)

    new_path = os.path.splitext(image_path)[0] + ".jpeg"
    img.save(new_path, "JPEG")

    img.close()

    os.remove(image_path)

def process_folder_and_convert_to_jpeg(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.gif', '.bmp', '.webp')):
                image_path = os.path.join(root, file)

                convert_to_jpeg(image_path)
                

def get_label_and_index(dataset_dir:str):

    data_path = pathlib.Path(dataset_dir)
    label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())

    label_index = dict((name,index) for index,name in enumerate(label_names))
    
    with open("./labels.json", "w") as outfile:
        outfile.write(json.dumps(label_index, indent=4))
    
    return label_names,label_index


def get_image_paths(dataset_dir:str):

    data_path = pathlib.Path(dataset_dir)
    all_image_paths = list(data_path.glob('*/*'))
    all_image_paths = [str(p) for p in all_image_paths]
    random.shuffle(all_image_paths)
    return all_image_paths


def load_and_process_from_path_label(image_path,image_label):
    print(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[128,128])
    # Décommenter cette ligne pour le model CNN et MLP
    image = tf.image.rgb_to_grayscale(image)
    image = image/255.0
    return image,image_label

    
def shuffle_image_label_ds(image_label_ds, image_count, batch_size, seed):
    image_label_ds = image_label_ds.shuffle(buffer_size=image_count, reshuffle_each_iteration=True, seed=seed)

    return image_label_ds.batch(batch_size)


def handle_dataset(data_path, batch_size, seed):

  label_names, label_index = get_label_and_index(data_path)

  image_paths = get_image_paths(data_path)
  image_count = len(image_paths)

  image_labels = [label_index[pathlib.Path(path).parent.name] for path in image_paths]

  paths_labels_ds = tf.data.Dataset.from_tensor_slices((image_paths,image_labels))

  image_label_ds = paths_labels_ds.map(load_and_process_from_path_label)

  image_label_shuffle_ds = shuffle_image_label_ds(image_label_ds, image_count, batch_size, seed)

  return image_label_shuffle_ds