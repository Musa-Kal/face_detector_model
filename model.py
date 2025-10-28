from tensorflow.keras import layers, models

class FaceBBoxModel:
    def __init__(self, input_size):
        self.model = None
        self.__build_model(input_size)

    def __build_model(self, input_size):
        # shaped like this cause why not 
        inputs = layers.Input(shape=input_size)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(5, activation='sigmoid')(x)
        self.model = models.Model(inputs, x)