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
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    local_zip = 'testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/testdata/')
    zip_ref.close()
    
                    #'/tmp/horse-or-human'
    TRAIN_DIR = './tmp/horse-or-human'
    TEST_DIR = './tmp/testdata'
                #'/tmp/testdata

    train_datagen = ImageDataGenerator(
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting.)
        rescale=1./255,horizontal_flip=True,vertical_flip=True,width_shift_range=(-1,1),height_shift_range=(-1,1),fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1. / 255)#Your Code here)

    train_generator = train_datagen.flow_from_directory('tmp/horse-or-human/', target_size=(300, 300),batch_size=20,class_mode='binary')
        #Your Code Here)

    validation_generator = validation_datagen.flow_from_directory('tmp/testdata/', target_size=(300, 300),batch_size=20,class_mode='binary')
        #Your Code Here)


    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here
        tf.keras.layers.Conv2D(64,3,input_shape = (300,300,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(256,(3,3), activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),       
        tf.keras.layers.Dense(32), 
        # This is the last layer. You should not change this code.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')#Your Code Here#)
    lr = ReduceLROnPlateau(monitor='val_loss',patience=15,factor=0.1,verbose=1) 
    es = EarlyStopping(monitor='val_loss',patience=30,mode='auto')

    model.fit(train_generator, epochs= 50,validation_data = validation_generator,validation_steps = len(validation_generator), callbacks=[es,lr])#Your Code Here#)

    results = model.evaluate(validation_generator)
    print("loss : ", results[0])
    print("acc : ", results[1])
    
    return model
    
    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("./Category3/mymodel.h5")
    