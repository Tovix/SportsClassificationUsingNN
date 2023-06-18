import os
import shutil
import tensorflow as tf
import directoryManipulation
import tester


class PreProcessor:
    """
    split the dataset and generates the Train , valid and test batches

    attributes:
    dataPath: the prediction list resulted from a model
    classes: the train classes of the dataset
    copyDataPath: the directory of the copied dataset
    directoryManipulator: object of the directoryManipulator class

    """

    def __init__(self, dataPath: str, classes: list) -> None:
        """
        preProcessor class initializer

        Parameters:
        :param dataPath: the prediction list resulted from a model
        :param classes: the train classes of the dataset

        Returns:
        :rtype: None
        """
        self.dataPath = dataPath
        self.classes = classes
        self.copyDataPath = ""
        self.directoryManipulator = directoryManipulation.DirectoryManipulator(self.dataPath, self.classes)

    def __initializeCopyData(self, ppTrainPath: str) -> None:
        """
        creates a copy of the original dataset

        Parameters:
        :param self: preProcessor class initializer
        :param ppTrainPath: the path of the copied dataset

        Returns:
        :rtype: None
        """
        self.directoryManipulator.initializeDirectory(self.dataPath, 'ppTrain')
        self.directoryManipulator.fillDirectory(self.dataPath, ppTrainPath, -1)

    def __createCopyData(self) -> None:
        """
        check if the data is copied or not

        Parameters:
        :param self: preProcessor class initializer

        Returns:
        :rtype: None
        """
        ppTrainPath = os.path.join(self.dataPath, 'ppTrain')
        if os.path.isdir(ppTrainPath) is True:
            shutil.rmtree(os.path.join(self.dataPath, 'ppTrain'))
            self.__initializeCopyData(ppTrainPath)
        else:
            self.__initializeCopyData(ppTrainPath)
        self.copyDataPath = ppTrainPath

    def splitData(self, trainSize: int, validationSize: int, testSize: int) -> None:
        """
        split the data into train, valid and test sets

        Parameters:
        :param self: preProcessor class initializer
        :param trainSize: the percentage of train data
        :param validationSize: the percentage of valid data
        :param testSize: the percentage of test data

        Returns:
        :rtype: None
        """
        self.__createCopyData()
        if os.path.isdir(os.path.join(self.dataPath, 'train')) is True:
            shutil.rmtree(os.path.join(self.dataPath, 'train'))
            shutil.rmtree(os.path.join(self.dataPath, 'valid'))
            shutil.rmtree(os.path.join(self.dataPath, 'test'))
            self.directoryManipulator.initializeClasses()
        else:
            self.directoryManipulator.initializeClasses()
        dataSize = len(os.listdir(self.copyDataPath))
        trainLimit = round(((trainSize * dataSize) / 100) / 6)
        validLimit = round(((validationSize * dataSize) / 100) / 6)
        testLimit = round(((testSize * dataSize) / 100) / 6)
        for cls in self.classes:
            destTrainPath = os.path.join(self.dataPath + '/train', cls)
            destValidPath = os.path.join(self.dataPath + '/valid', cls)
            destTestPath = os.path.join(self.dataPath + '/test', cls)
            self.directoryManipulator.fillDirectory(self.copyDataPath, destTrainPath, trainLimit, cls)
            self.directoryManipulator.fillDirectory(self.copyDataPath, destValidPath, validLimit, cls)
            self.directoryManipulator.fillDirectory(self.copyDataPath, destTestPath, testLimit, cls)

    def generateTVTBatches(self, imgSize: tuple, trainBatchSize: int,
                           validationBatchSize: int, testBatchSize: int) -> list:
        """
        split the data into train, valid and test sets

        Parameters:
        :param self: preProcessor class initializer
        :param imgSize: the size of img in the batch
        :param trainBatchSize: the size of train data batch
        :param validationBatchSize: the size of valid data batch
        :param testBatchSize: the size of test data batch

        Returns:
        :rtype: list: list of TVT batches
        """
        trainPath = os.path.join(self.dataPath, 'train')
        validPath = os.path.join(self.dataPath, 'valid')
        testPath = os.path.join(self.dataPath, 'test')
        trainBatches = tf.keras.preprocessing.image.ImageDataGenerator(). \
            flow_from_directory(directory=trainPath, target_size=(imgSize[0], imgSize[1]),
                                classes=self.classes, batch_size=trainBatchSize)
        validBatches = tf.keras.preprocessing.image.ImageDataGenerator(). \
            flow_from_directory(directory=validPath, target_size=(imgSize[0], imgSize[1]),
                                classes=self.classes, batch_size=validationBatchSize)
        testBatches = tf.keras.preprocessing.image.ImageDataGenerator(). \
            flow_from_directory(directory=testPath, target_size=(imgSize[0], imgSize[1]),
                                classes=self.classes, batch_size=testBatchSize)
        return [trainBatches, validBatches, testBatches]


path = "N:/Downloads/SportsClassification/Train/"
dataClasses = ['basketball', 'football', 'rowing', 'swimming', 'tennis', 'yoga']
p1 = PreProcessor(path, dataClasses)
p1.splitData(90, 5, 5)
batches = p1.generateTVTBatches((299, 299), 1, 1, 1)
# VGG16 = models.BasicCNNModel(batches, (299, 299, 3), "VGG16")
# 
# print(VGG16.model.summary())
# VGG16.buildModel('adam', 'categorical_crossentropy', ['accuracy'], 1, 2)

loadedModel = r"N:\Downloads\SportsClassification\incModel"
model = tf.keras.models.load_model(loadedModel)
print(model.summary())

tstPath = r"N:\Downloads\SportsClassification\TestS"
tester = tester.ModelTester()
incRes = tester.testModel(tstPath, (299, 299), 20, model, True, batches)
