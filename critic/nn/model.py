from keras.layers import Dense, Dropout, BatchNormalization
from keras import Sequential

nn_model = Sequential()
dim = 300
nn_model.add(Dense(256, activation="relu", input_dim=dim))
nn_model.add(BatchNormalization())
nn_model.add(Dropout(0.5))
nn_model.add(Dense(128, activation="relu"))
nn_model.add(BatchNormalization())
nn_model.add(Dense(3, activation="softmax"))
nn_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
