import tensorflow as tf
from tensorflow.keras import layers, models
import datetime

from displayReport import display_report

def create_model(classes = 8):
    
    model = models.Sequential([
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(classes)
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

    # score = model.evaluate(test_dataset, verbose=0, steps=steps_per_epoch )
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    
    ###################################################################

    display_report(train_dataset, model, "CNN", "Train")
    
    ###################################################################

    display_report(validation_dataset, model, "CNN", "Validation")
    
    ###################################################################

    display_report(test_dataset, model, "CNN", "Test")