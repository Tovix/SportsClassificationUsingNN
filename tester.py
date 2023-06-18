import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from directoryManipulation import DirectoryManipulator


class ModelTester:
    """
    tests a given model and generates results compared to a test data
    """
    @staticmethod
    def __initializeTestFolder(testPath: str, destPath: str) -> None:
        """
        creates a test directory which will be used by the test generator

        Parameters:
        :param testPath: the path of the test directory
        :param destPath: the path in which the test directory will be created

        Returns:
        :rtype: None
        """
        DirectoryManipulator.initializeDirectory(testPath, 'test')
        DirectoryManipulator.fillDirectory(testPath, destPath, -1)

    @staticmethod
    def __checkPredictions(predictions: list, unlabeledTestBatches: tf.keras.preprocessing.image.ImageDataGenerator) -> list:
        """
        matches the prediction names with test results of the model and generates the results CSV file

        Parameters:
        :param predictions: the predictions list
        :param unlabeledTestBatches: the image generator of the test path
        Returns:
        :rtype: list: the result of the model predictions
        """
        normalizePredictions = []
        fileNameList = []
        fileCounter = 0
        print(predictions)
        print(len(predictions))
        for prediction in predictions:
            fileName = unlabeledTestBatches.filenames[fileCounter]
            fileNameList.append(fileName.split('\\')[1])
            normalizePredictions.append(np.argmax(prediction))
            fileCounter += 1
        testList = np.dstack((np.array(fileNameList), np.array(normalizePredictions))).reshape(-1, 2)
        DirectoryManipulator.outputResult(testList=testList)
        return testList

    def testModel(self, testPath: str, imgSize: tuple, batchSize: int, model: tf.keras.models, isMultiLayer: bool, batches: list) -> list:
        """
        tests the model

        Parameters:
        :param testPath: the path of the test
        :param imgSize: the size of the test image size
        :param batchSize: the batch size of test generator

        Returns:
        :rtype: list: the result of the model predictions
        """
        destPath = os.path.join(testPath, 'test')
        if os.path.isdir(destPath) is True:
            shutil.rmtree(destPath)
            self.__initializeTestFolder(testPath, destPath)
        else:
            self.__initializeTestFolder(testPath, destPath)
        unlabeledTestBatches = tf.keras.preprocessing.image.ImageDataGenerator(). \
            flow_from_directory(directory=testPath, target_size=imgSize, batch_size=batchSize, shuffle=False)
        predictions = model.predict(unlabeledTestBatches)
        testAccuracy = model.evaluate(x=batches[2])
        results = []
        results.append(testAccuracy)
        if isMultiLayer:
            for layerPrediction in predictions:
                results.append(self.__checkPredictions(layerPrediction, unlabeledTestBatches))
            return results
        else:
            return [testAccuracy, self.__checkPredictions(predictions, unlabeledTestBatches)]

    @staticmethod
    def modelCSVAccuracy(testFile: str, testedFile: str) -> float:
        """
        return the accuracy of the model based on csv files

        Parameters:
        :param testFile: the name of the test file
        :param testedFile: the name of the tested file

        Returns:
        :rtype: the model accuracy
        """
        df1 = pd.read_csv(testFile)
        df1 = df1.sort_values('image_name')
        df2 = pd.read_csv(testedFile)
        df2 = df2.sort_values('image_name')
        accuracy = 0
        for i in df1.index:
            x = df1.iloc[i, 1]
            y = df2.iloc[i, 1]
            if x == y:
                accuracy += 1
        return accuracy * 100 / 688

    @staticmethod
    def modelAccuracy(testPath: str, batchSize: int, noOfClasses: int, imgSize: tuple, model: tf.keras.models) -> float:
        """
        return the accuracy of the model

        Parameters:
        :param testPath: the path of the test
        :param batchSize: the size of the batch
        :param noOfClasses: the number of classes of the model
        :param imgSize: the size of the image
        :param model: the model that will be tested

        Returns:
        :rtype: float: the model accuracy
        """
        data = tf.keras.utils.image_dataset_from_directory(testPath, batch_size=batchSize, image_size=imgSize)
        dataIter = data.as_numpy_iterator()
        accuracy = 0

        def convert(batchS, nClasses, btch):
            array = np.zeros((batchS, nClasses))
            for j in range(len(btch[1])):
                array[j, btch[1][j]] = 1
            return array

        for i in range(len(data) - 1):
            batch = dataIter.next()
            result = model.evaluate(x=batch[0], y=convert(batchSize, noOfClasses, batch))
            accuracy += result[1]
        return accuracy / len(data)
