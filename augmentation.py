import cv2
import random
import os.path
import numpy as np
from tqdm import tqdm
import tensorflow as tf


class Augmentor:
    """
     Augments dataset with various options

     attributes:
     :param sourcePath: the source of the dataset to be augmented
     :param destPath: the destination where the augmented pictures will be saved

    """
    def __init__(self, sourcePath: str, destPath: str) -> None:
        """
        Augmentor class initializer

        Parameters:
        :param sourcePath: the source of the dataset to be augmented
        :param destPath: the destination where the augmented pictures will be saved

        Returns:
        :rtype: None
        """
        self.sourcePath = sourcePath
        self.destPath = destPath

    @staticmethod
    def __setAugOptions(rotationRange: float, wsRange: float, hsRange: float,
                        shearRange: float, zoomRange: float, hFlip: bool, fillMode: str) -> tf.keras.preprocessing.image.ImageDataGenerator:
        """
        set the options of the produced augmented image

        Parameters:
        :param rotationRange: the range of rotation of the image
        :param wsRange: the wsRange of the image
        :param hsRange: the hsRange of the image
        :param shearRange: the shear range of the image
        :param zoomRange: the zoom range of the image
        :param hFlip: the horizontal range of the image
        :param fillMode: the fill mode of the image

        Returns:
        :rtype: tf.keras.preprocessing.ImageDataGenerator:
        """
        augDataGenerator = tf.keras.preprocessing.image. \
            ImageDataGenerator(rotation_range=rotationRange, width_shift_range=wsRange,
                               height_shift_range=hsRange, shear_range=shearRange, zoom_range=zoomRange,
                               horizontal_flip=hFlip, fill_mode=fillMode)
        return augDataGenerator

    def __generateAugImages(self, imageGenerator: tf.keras.preprocessing.image.ImageDataGenerator, batchSize: int, saveFormat: str, batchNumber: int) -> None:
        """
        generates augmented images

        Parameters:
        :param imageGenerator: the image generator of the augmented images
        :param batchSize: the size of the produced batch
        :param saveFormat: the format of the images that will be saved
        :param batchNumber: the batch production number

        Returns:
        :rtype: None
        """
        path = os.listdir(self.sourcePath)
        for imgName in tqdm(path):
            seed = random.randint(100000, 999999)
            if imgName != 'train' and imgName != 'test' and imgName != 'valid' and imgName != "ppTrain":
                if "aug" not in imgName:
                    img = cv2.imread(f'{self.sourcePath}/{imgName}', 1)
                    npArr = np.array(img)
                    augImgIterator = imageGenerator.flow(x=np.expand_dims(npArr, 0), batch_size=batchSize, save_format=saveFormat)
                    augImg = next(augImgIterator)
                    cv2.imwrite(f'{self.sourcePath}/aug{batchNumber * seed}_{imgName}', augImg[0, :, :, :])

    def generateNAugBatches(self, noOfBatches: int, saveFormat: str, batchSize: int) -> None:
        """
        generates N batches of augmented images

        Parameters:
        :param noOfBatches: the image generator of the augmented images
        :param batchSize: the size of the produced batch
        :param saveFormat: the format of the images that will be saved

        Returns:
        :rtype: None
        """
        breaker = 0
        while breaker < noOfBatches:
            x1 = round(random.uniform(0.1, 0.9), 1)
            x2 = round(random.uniform(0.1, 0.9), 1)
            x3 = round(random.uniform(0.1, 0.9), 1)
            x4 = round(random.uniform(0.1, 0.9), 1)
            x5 = round(random.uniform(0.1, 0.9), 1)
            imgGen = self.__setAugOptions(x1, x2, x3, x4, x5, True, "reflect")
            tqdm(self.__generateAugImages(imgGen, batchSize, saveFormat, breaker + 1))
            breaker += 1


sPath = r"N:\Downloads\SportsClassification\Train"
dPath = r"N:\Downloads\SportsClassification\Train\aug"
Augmentor = Augmentor(sourcePath=sPath, destPath=dPath)

Augmentor.generateNAugBatches(3, 'png', 1)
