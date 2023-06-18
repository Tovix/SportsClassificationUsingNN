import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, MaxPool2D, Input, GlobalAveragePooling2D, \
    AveragePooling2D, Dropout, Activation, BatchNormalization, concatenate


class BasicCNNModel:
    """
    BasicCNNModel with various options

    attributes:
    dataBatches: the optimizer used in the Model
    inputShape: the loss function of the model
    layers: the layers of the model
    """

    def __init__(self, dataBatches: list, inputShape: tuple, Mtype: str = "BasicCNNModel") -> None:
        """
        BasicCNNModel initializer

        Parameters:
        :param self: BasicCNNModel instance
        :param dataBatches: the optimizer used in the Model
        :param inputShape: the loss function of the model
        :param Mtype: determines the type of the model
        Returns:
        :rtype: None
        """
        if Mtype == "VGG16":
            self.layers = [
                Conv2D(input_shape=inputShape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
                Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
                MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
                Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
                MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
                Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
                Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
                MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
                Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
                Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
                MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
                Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
                Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
                MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                Flatten(),
                Dense(units=4096, activation="relu"),
                Dense(units=4096, activation="relu"),
                Flatten(),
                Dense(units=6, activation="softmax")
            ]
        else:
            self.layers = \
                [
                    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=inputShape),
                    tf.keras.layers.BatchNormalization(),
                    MaxPooling2D(pool_size=(2, 2), strides=2),
                    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
                    MaxPooling2D(pool_size=(2, 2), strides=2),
                    Flatten(),
                    Dense(units=6, activation="softmax")
                ]
        self.model = tf.keras.Sequential(self.layers)
        self.batches = dataBatches

    def buildModel(self, optimizer: tf.keras.optimizers, loss: str, metrics: list,
                   epochs: int, verbose: int) -> list:
        """
        compiles and fits the CNN model

        Parameters:
        :param self: BasicCNNModel initializer
        :param optimizer: the optimizer used in the Model
        :param loss: the loss function of the model
        :param metrics: the metrics used as the output
        :param epochs: the number of epochs of training
        :param verbose: the degree clarity of the output

        Returns:
        :rtype: list: the result of the model evaluation and it's history
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                         mode="min", patience=5,
                                                         restore_best_weights=True)
        self.model.fit(x=self.batches[0], validation_data=self.batches[1], epochs=epochs, verbose=verbose, callbacks=[earlyStopping])
        results = self.model.evaluate(x=self.batches[2])
        return results


class InceptionV3:
    def __init__(self, inputShape: tuple, noOfClasses: int, dataBatches: list) -> None:
        self.inputShape = inputShape
        self.noOfClasses = noOfClasses
        self.model = self.__BuildInceptionV3Layers()
        self.dataBatches = dataBatches

    def __BuildInceptionV3Layers(self):
        input_layer = Input(shape=self.inputShape)
        x = self.StemBlock(input_layer)
        x = self.__InceptionBlockA(prev_layer=x, nbr_kernels=32)
        x = self.__InceptionBlockA(prev_layer=x, nbr_kernels=64)
        x = self.__InceptionBlockA(prev_layer=x, nbr_kernels=64)
        x = self.__ReductionBlockA(prev_layer=x)
        x = self.__InceptionBlockB(prev_layer=x, nbr_kernels=128)
        x = self.__InceptionBlockB(prev_layer=x, nbr_kernels=160)
        x = self.__InceptionBlockB(prev_layer=x, nbr_kernels=160)
        x = self.__InceptionBlockB(prev_layer=x, nbr_kernels=192)
        x = self.__ReductionBlockB(prev_layer=x)
        x = self.__InceptionBlockC(prev_layer=x)
        x = self.__InceptionBlockC(prev_layer=x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=2048, activation='relu')(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=self.noOfClasses, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=x, name='Inception-V3')
        return model

    @staticmethod
    def __convWithBatchNormal(prev_layer, nbr_kernels, filter_Size, strides=(1, 1), padding='same'):
        x = Conv2D(filters=nbr_kernels, kernel_size=filter_Size, strides=strides, padding=padding)(prev_layer)
        x = BatchNormalization(axis=3)(x)
        x = Activation(activation='relu')(x)
        return x

    def StemBlock(self, prev_layer):
        x = self.__convWithBatchNormal(prev_layer, nbr_kernels=32, filter_Size=(3, 3), strides=(2, 2))
        x = self.__convWithBatchNormal(x, nbr_kernels=32, filter_Size=(3, 3))
        x = self.__convWithBatchNormal(x, nbr_kernels=64, filter_Size=(3, 3))
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = self.__convWithBatchNormal(x, nbr_kernels=80, filter_Size=(1, 1))
        x = self.__convWithBatchNormal(x, nbr_kernels=192, filter_Size=(3, 3))
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        return x

    def __InceptionBlockA(self, prev_layer, nbr_kernels):
        branch1 = self.__convWithBatchNormal(prev_layer, nbr_kernels=64, filter_Size=(1, 1))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=96, filter_Size=(3, 3))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=96, filter_Size=(3, 3))
        branch2 = self.__convWithBatchNormal(prev_layer, nbr_kernels=48, filter_Size=(1, 1))
        branch2 = self.__convWithBatchNormal(branch2, nbr_kernels=64, filter_Size=(3, 3))  # may be 3*3
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_layer)
        branch3 = self.__convWithBatchNormal(branch3, nbr_kernels=nbr_kernels, filter_Size=(1, 1))
        branch4 = self.__convWithBatchNormal(prev_layer, nbr_kernels=64, filter_Size=(1, 1))
        output = concatenate([branch1, branch2, branch3, branch4], axis=3)
        return output

    def __InceptionBlockB(self, prev_layer, nbr_kernels):
        branch1 = self.__convWithBatchNormal(prev_layer, nbr_kernels=nbr_kernels, filter_Size=(1, 1))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=nbr_kernels, filter_Size=(7, 1))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=nbr_kernels, filter_Size=(1, 7))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=nbr_kernels, filter_Size=(7, 1))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=192, filter_Size=(1, 7))
        branch2 = self.__convWithBatchNormal(prev_layer, nbr_kernels=nbr_kernels, filter_Size=(1, 1))
        branch2 = self.__convWithBatchNormal(branch2, nbr_kernels=nbr_kernels, filter_Size=(1, 7))
        branch2 = self.__convWithBatchNormal(branch2, nbr_kernels=192, filter_Size=(7, 1))
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_layer)
        branch3 = self.__convWithBatchNormal(branch3, nbr_kernels=192, filter_Size=(1, 1))
        branch4 = self.__convWithBatchNormal(prev_layer, nbr_kernels=192, filter_Size=(1, 1))
        output = concatenate([branch1, branch2, branch3, branch4], axis=3)
        return output

    def __InceptionBlockC(self, prev_layer):
        branch1 = self.__convWithBatchNormal(prev_layer, nbr_kernels=448, filter_Size=(1, 1))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=384, filter_Size=(3, 3))
        branch1_1 = self.__convWithBatchNormal(branch1, nbr_kernels=384, filter_Size=(1, 3))
        branch1_2 = self.__convWithBatchNormal(branch1, nbr_kernels=384, filter_Size=(3, 1))
        branch1 = concatenate([branch1_1, branch1_2], axis=3)
        branch2 = self.__convWithBatchNormal(prev_layer, nbr_kernels=384, filter_Size=(1, 1))
        branch2_1 = self.__convWithBatchNormal(branch2, nbr_kernels=384, filter_Size=(1, 3))
        branch2_2 = self.__convWithBatchNormal(branch2, nbr_kernels=384, filter_Size=(3, 1))
        branch2 = concatenate([branch2_1, branch2_2], axis=3)
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_layer)
        branch3 = self.__convWithBatchNormal(branch3, nbr_kernels=192, filter_Size=(1, 1))
        branch4 = self.__convWithBatchNormal(prev_layer, nbr_kernels=320, filter_Size=(1, 1))
        output = concatenate([branch1, branch2, branch3, branch4], axis=3)
        return output

    def __ReductionBlockA(self, prev_layer):
        branch1 = self.__convWithBatchNormal(prev_layer, nbr_kernels=64, filter_Size=(1, 1))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=96, filter_Size=(3, 3))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=96, filter_Size=(3, 3), strides=(2, 2))
        branch2 = self.__convWithBatchNormal(prev_layer, nbr_kernels=384, filter_Size=(3, 3), strides=(2, 2))
        branch3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(prev_layer)
        output = concatenate([branch1, branch2, branch3], axis=3)
        return output

    def __ReductionBlockB(self, prev_layer):
        branch1 = self.__convWithBatchNormal(prev_layer, nbr_kernels=192, filter_Size=(1, 1))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=192, filter_Size=(1, 7))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=192, filter_Size=(7, 1))
        branch1 = self.__convWithBatchNormal(branch1, nbr_kernels=192, filter_Size=(3, 3), strides=(2, 2), padding='valid')
        branch2 = self.__convWithBatchNormal(prev_layer, nbr_kernels=192, filter_Size=(1, 1))
        branch2 = self.__convWithBatchNormal(branch2, nbr_kernels=320, filter_Size=(3, 3), strides=(2, 2), padding='valid')
        branch3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(prev_layer)
        output = concatenate([branch1, branch2, branch3], axis=3)
        return output

    def buildModel(self, optimizer: tf.keras.optimizers, loss: str, metrics: list,
                   epochs: int, verbose: int, validFreq: int) -> list:
        """
        compiles and fits the inceptionV3 model

        Parameters:
        :param self: BasicCNNModel initializer
        :param optimizer: the optimizer used in the Model
        :param loss: the loss function of the model
        :param metrics: the metrics used as the output
        :param epochs: the number of epochs of training
        :param verbose: the degree clarity of the output
        :param validFreq: the validation frequency of the model

        Returns:
        :rtype: list: the result of the model evaluation
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.fit(x=self.dataBatches[0], validation_data=self.dataBatches[1], epochs=epochs, verbose=verbose, validation_freq=validFreq)
        results = self.model.evaluate(x=self.dataBatches[2])
        return results
