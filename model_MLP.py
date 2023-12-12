import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import datetime

from displayReport import display_report

def create_model(classes = 8):
    
    model = models.Sequential([

        layers.Flatten(input_shape=(128, 128, 1)),

        layers.Dense(64*64, activation='relu'),
        layers.Dense(16*16, activation='relu'),
        layers.Dense(8*8, activation='relu'),

        layers.Dense(classes, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    return model

def save_and_train_model(model_path, log_dir_base, train_dataset, validation_dataset, test_dataset):
    
    model = create_model()
    
    log_dir = log_dir_base + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="accuracy",
            save_best_only = True,
        ),
    ]

    model.fit(train_dataset, epochs=10, validation_data=validation_dataset, callbacks=[my_callbacks])
    
    ###################################################################

    display_report(train_dataset, model, "MLP", "Train")
    
    ###################################################################

    display_report(validation_dataset, model, "MLP", "Validation")
    
    ###################################################################

    display_report(test_dataset, model, "MLP", "Test")
