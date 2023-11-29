import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import datetime

def create_model():
    
    model = models.Sequential([
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)), #卷积层1，卷积核3*3
        layers.MaxPooling2D((2, 2)),                   #池化层1，2*2采样
        layers.Conv2D(256, (3, 3), activation='relu'),  #卷积层2，卷积核3*3
        layers.MaxPooling2D((2, 2)),                   #池化层2，2*2采样
        layers.Conv2D(256, (3, 3), activation='relu'),  #卷积层3，卷积核3*3

        layers.Flatten(),                      #Flatten层，连接卷积层与全连接层
        layers.Dense(256, activation='relu'),   #全连接层，特征进一步提取
        layers.Dense(8)                       #输出层，输出预期结果
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    return model

def save_and_train_model(model_path, log_dir_base, train_dataset, validation_dataset, test_dataset, steps_per_epoch):
    
    model = create_model()
    
    log_dir = log_dir_base + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)

    my_callbacks = [
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     # patience=5,
        #     restore_best_weights=True,
        #     # start_from_epoch=3,
        # ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="accuracy",
            save_best_only = True,
        ),
        # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    model.fit(train_dataset, epochs=10, validation_data=validation_dataset, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch, callbacks=[my_callbacks])
    # steps_per_epoch * batch_size = number_of_rows_in_train_data

    score = model.evaluate(test_dataset, verbose=0, steps=steps_per_epoch )
    print('Test score:', score[0])
    print('Test accuracy:', score[1])