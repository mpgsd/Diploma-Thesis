import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


#DATA_PATH = "d:\\ΔΙπλωματική\\data_2.json"

DATA_PATH = "E:\\ΔΙπλωματική\\data_2.json"

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Combined MFCC and Mel Spectrogram inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    mfccs = np.array(data["mfcc"])
    mel_spectrograms = np.array(data["mel_spectrogram"])
    y = np.array(data["labels"])

    # Ensure the dimensions match for concatenation
    if mfccs.shape[0] != mel_spectrograms.shape[0]:
        raise ValueError("Mismatch in number of samples between MFCC and Mel Spectrogram data")

    # Concatenate MFCCs and Mel Spectrograms along the feature axis
    X = np.concatenate((mfccs, mel_spectrograms), axis=2)  # Assuming the features are along axis 2

    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0 , test_size=test_size, stratify=y)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,  random_state=0 , test_size=validation_size, stratify=y_train)
    # can also  use print(X_validation) for centainty
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates RNN-LSTM model

    input_shape: Shape of input set
    """

    
    model = tf.keras.Sequential() 

    # 2 LSTM layers
    model.add(tf.keras.layers.LSTM(130, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(130, return_sequences=True))
    model.add(tf.keras.layers.LSTM(130))
    # dense layer
    model.add(tf.keras.layers.Dense(130, activation='relu'))
    # can also use model.add(tf.keras.layers.Dropout(0.3))

    # output layer
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    return model


if __name__ == "__main__":

    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2]) # example 126, 13+128=141  two numbers that show time steps and feature number per time step
    model = build_model(input_shape)

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=50)
    mean_train_acc = np.mean(history.history['accuracy'])
    mean_val_acc = np.mean(history.history['val_accuracy'])
    print(f"Mean Training Accuracy: {mean_train_acc:.4f}, Mean Validation Accuracy: {mean_val_acc:.4f}")
    
    # plot accuracy/error for training and validation
    plot_history(history)
    
    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f} ")
