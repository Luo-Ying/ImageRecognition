import datetime

import numpy as np
import tensorflow as tf
from displayReport import display_report

def create_model(isPretrained, classes = 8):
    
    if isPretrained:
        model = models.Sequential()

        model.add(tf.keras.applications.resnet50.ResNet50(include_top=False, pooling='avg', weights="imagenet"))
        model.add(layers.Flatten())
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(classes, activation='softmax'))

        model.layers[0].trainable = False
        
    else:
        model = tf.keras.applications.resnet50.ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=classes,
            classifier_activation="softmax",
        )

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    return model

def save_and_train_model(model_path, log_dir_base, train_dataset, validation_dataset, test_dataset, isPretrained = False):
    
    model = create_model(isPretrained)
    
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

    display_report(train_dataset, model, "ResNet50", "Train")
    
    ###################################################################

    display_report(validation_dataset, model, "ResNet50", "Validation")
    
    ###################################################################

    display_report(test_dataset, model, "ResNet50", "Test")