import os
import shutil
import random
import pandas as pd
from tqdm import tqdm


class DirectoryManipulator:
    """
        manipulates directories and files

        attributes:
        :param dataPath: the general path
        :param classes: the list of training classes

    """
    def __init__(self, dataPath: str, classes: list) -> None:
        """
        DirectoryManipulator class initializer

        Parameters:
        :param dataPath: the general path
        :param classes: the list of training classes

        Returns:
        :rtype: None
        """
        self.dataPath = dataPath
        self.classes = classes

    @staticmethod
    def initializeDirectory(specPath: str, directoryName: str) -> None:
        """
        creates a new directory at the specified path

        Parameters:
        :param specPath: the path in which the directory will be created
        :param directoryName: the name of the created directory

        Returns:
        :rtype: None
        """
        trainPath = os.path.join(specPath, directoryName)
        os.mkdir(trainPath)

    @staticmethod
    def fillDirectory(sourcePath: str, destPath: str, size: int, className: str = "", move: bool = True) -> None:
        """
        fill destination directory from source directory

        Parameters:
        :param sourcePath: the path of directory that contains the data that will be transferred
        :param destPath: the path of directory that will be filled
        :param size: the number of files
        :param className: the className that will be used to move the file to it's corresponding class directory
        :param move: indicates whether the file will be moved or copied

        Returns:
        :rtype: None
        """
        files = os.listdir(sourcePath)
        if size == -1:
            random.shuffle(files)
            for fileName in files:
                srcPath = os.path.join(sourcePath, fileName)
                if os.path.isfile(srcPath):
                    shutil.copy(srcPath, destPath)
        else:
            stopCounter = 0
            random.shuffle(files)

            for fileName in tqdm(files):
                srcPath = os.path.join(sourcePath, fileName)
                if stopCounter == size:
                    break
                if className.lower() in fileName.lower():
                    if move:
                        shutil.move(srcPath, destPath)
                    else:
                        shutil.copy(srcPath, destPath)
                    stopCounter += 1

    def initializeTVT(self) -> None:
        """
        creates the train, valid and test directories in the general path

        Parameters:
        :param self: DirectoryManipulator class

        Returns:
        :rtype: None
        """
        self.initializeDirectory(self.dataPath, "train")
        self.initializeDirectory(self.dataPath, "valid")
        self.initializeDirectory(self.dataPath, "test")

    def initializeClasses(self) -> None:
        """
        creates the classes directories in each of the TVT directories

        Parameters:
        :param self: DirectoryManipulator class

        Returns:
        :rtype: None
        """
        self.initializeTVT()
        for cls in self.classes:
            self.initializeDirectory(self.dataPath + "/train", cls)
            self.initializeDirectory(self.dataPath + "/valid", cls)
            self.initializeDirectory(self.dataPath + "/test", cls)

    @staticmethod
    def outputResult(testList: list) -> None:
        """
        creates the CSV result file

        Parameters:
        :param testList: the prediction list resulted from a model

        Returns:
        :rtype: None
        """
        seed = random.randint(100000, 999999)
        dataFrame = pd.DataFrame(testList)
        dataFrame.columns = ['image_name', "label"]
        dataFrame.to_csv(f"res{seed}.csv", index=False)