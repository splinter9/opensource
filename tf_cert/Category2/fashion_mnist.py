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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D,Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import time


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    out_node = len(np.unique(y_train))

    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Conv1D(150, 2, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Flatten())
    # model.add(LSTM(150,activation = 'relu', input_shape = (x_train.shape[1],x_train.shape[2])))
    model.add(Dense(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_node, activation='softmax'))

    opt = "adam"
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])  # metrics=['accuracy'] 영향을 미치지 않는다
    ########################################################################
    # model.compile(loss = 'mse', optimizer = 'adam')
    start = time.time()
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    import datetime
    epoch = 100
    patience_num = 5
    date = datetime.datetime.now()
    datetime = date.strftime("%m%d_%H%M")
    filepath = "./_ModelCheckPoint/"
    filename = '{epoch:04d}-{val_loss:4f}.hdf5'  # filepath + datetime
    model_path = "".join([filepath, 'k35_cfar10_', datetime, "_", filename])
    es = EarlyStopping(monitor='val_loss', patience=patience_num, mode='auto', restore_best_weights=True)
    # mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
    hist = model.fit(x_train, y_train, epochs=epoch, validation_split=0.2, callbacks=[es], batch_size=500)
    end = time.time() - start
    print('시간 : ', round(end, 2), '초')
    ########################################################################

    # 4 평가예측
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)
    print("loss : ", loss[0])
    print("accuracy : ", loss[1])

    # YOUR CODE HERE
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

'''
loss :  0.3532934784889221
accuracy :  0.8830000162124634
'''