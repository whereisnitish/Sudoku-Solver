import numpy as np
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load the trained model
cnn_model11 = load_model('cnn_model11.h5')
optimizer = tf.keras.optimizers.Adam(0.001)
cnn_model11.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Define a function to make predictions for Sudoku puzzles
def predict_sudoku(sudoku_input):

    sudoku_input = np.array(list(map(int,list(sudoku_input)))).reshape(9,9)

    # Preprocess the input data
    normalized_input = sudoku_input / 9.0
    normalized_input -= .5
    input_data = normalized_input.reshape(1, 9, 9, 1)
    predictions = cnn_model11.predict(input_data)
    sudoku_output = predictions.argmax(-1).reshape((9, 9)) + 1

    return sudoku_output


