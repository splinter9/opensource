# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def solution_model():
    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    # urllib.request.urlretrieve(url, 'rps.zip')
    # local_zip = 'rps.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('tmp/')
    # zip_ref.close()

    #1. 데이터
    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
    # YOUR CODE HERE
        rescale=1/255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1)

    train_generator = training_datagen.flow_from_directory(# YOUR CODE HERE
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=64,
        class_mode='categorical',
        subset='training')

    valid_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=64,
        subset='validation'
    )
    #2. 모델
    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),     # (74, 74, 64)
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (6, 6), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  patience=3,
                                  factor=0.5,
                                  verbose=1)
    checkpoint = ModelCheckpoint("checkpoint.ckpt",
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
    model.fit(train_generator,
              steps_per_epoch=len(train_generator),
              validation_data=(valid_generator),
              validation_steps=len(valid_generator),
              epochs=21,
              verbose=1,
              callbacks=[earlystopping, reduce_lr, checkpoint]
              )
    results = model.evaluate(validation_generator)
    print("loss : ", results[0])
    print("acc : ", results[1])
    
    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")