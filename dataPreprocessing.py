from PIL import Image
import tensorflow  as tf
import os
import pathlib
import random

# dataset_path = './data_set/skin-disease-datasaet/'
# BATCH_SIZE = 64

def convert_to_jpeg(image_path):
    # 打开图像文件
    img = Image.open(image_path)

    # 将图像保存为JPEG格式
    new_path = os.path.splitext(image_path)[0] + ".jpeg"
    img.save(new_path, "JPEG")

    # 关闭图像文件
    img.close()

    # 删除原始文件
    os.remove(image_path)

def process_folder_and_convert_to_jpeg(folder_path):
    # 遍历文件夹内所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为图像文件
            if file.lower().endswith(('.png', '.jpg', '.gif', '.bmp', '.webp')):
                # 构建图像文件的完整路径
                image_path = os.path.join(root, file)

                # 将图像文件转换为JPEG格式并替换原文件
                convert_to_jpeg(image_path)
                
# get class name and it's index
def get_label_and_index(dataset_dir:str):

    data_path = pathlib.Path(dataset_dir)
    label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())

    label_index = dict((name,index) for index,name in enumerate(label_names))
    return label_names,label_index

# Get paths of all files
def get_image_paths(dataset_dir:str):

    data_path = pathlib.Path(dataset_dir)
    all_image_paths = list(data_path.glob('*/*'))
    all_image_paths = [str(p) for p in all_image_paths]
    random.shuffle(all_image_paths)
    return all_image_paths

def load_and_process_from_path_label(image_path,image_label):
    print(image_path)
    image = tf.io.read_file(image_path) # 读取图像文件
    image = tf.image.decode_jpeg(image,channels=3)  # 将JPEG图像解码为具有3个通道（RGB）的张量
    image = tf.image.resize(image,[128,128])  # 将图像调整大小为（128, 128）
    image = image/255.0    # 将像素值归一化到范围[0, 1]
    return image,image_label

def shuffle_image_label_ds(image_label_ds, image_count, batch_size, seed):
    # image_label_ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    # image_label_ds = image_label_ds.shuffle(buffer_size=image_count, reshuffle_each_iteration=True).repeat()
    image_label_ds = image_label_ds.shuffle(buffer_size=image_count, reshuffle_each_iteration=True, seed=seed).repeat()

    return image_label_ds.batch(batch_size)

    # dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size, count, seed))
    # produces the same output as
    # dataset.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True).repeat(count)
    
# def shuffle_image_label_ds(image_label_ds, image_count, batch_size):
#     # image_label_ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
#     image_label_ds = image_label_ds.shuffle(buffer_size=image_count, reshuffle_each_iteration=True).repeat()

#     return image_label_ds.batch(batch_size)

def handle_dataset(data_path, batch_size, seed):

  # process_folder_and_convert_to_jpeg(data_path)

  label_names, label_index = get_label_and_index(data_path)

  image_paths = get_image_paths(data_path)
  image_count = len(image_paths)

  # print(label_index)
  # print(image_count)

  image_labels = [label_index[pathlib.Path(path).parent.name] for path in image_paths]
  # for image, label in zip(image_paths, image_labels):
  #     print(image, ' --->  ', label)


  #创建图片路径及其数字标签的dataset
  paths_labels_ds = tf.data.Dataset.from_tensor_slices((image_paths,image_labels))

  # print(paths_labels_ds)

  image_label_ds = paths_labels_ds.map(load_and_process_from_path_label)
  # print('\n image_label_ds \n',image_label_ds)

  # 对数据集进行洗牌
  image_label_shuffle_ds = shuffle_image_label_ds(image_label_ds, image_count, batch_size, seed)
  # print('\n image_label_shuffle_ds \n',image_label_shuffle_ds)

  return image_label_shuffle_ds