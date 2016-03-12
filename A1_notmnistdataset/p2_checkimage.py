import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math
import numpy as np

class CheckImage:
    def checkImageinPicle(self, randindexes):
        with open('notMNIST_large/J.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
        self.dispImages(dataset[randindexes], 2)
        return
    def checkOriginalImage(self, randindexes):
        with open('notMNIST_large/J_filepath.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
        self.dispImages_filepath(dataset[randindexes], 1)
        return
    def checknotMNIST_2_pickle(self,randindexes):
        with open('notMNIST_3.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
            self.dispImages_filepath(dataset['train_filepaths'][randindexes], 1)
            self.dispImages(dataset['train_dataset'][randindexes], 2)
#             self.dispImages_filepath(dataset['valid_filepaths'][randindexes], 1)
#             self.dispImages(dataset['valid_dataset'][randindexes], 2)
#         self.dispImages_filepath(dataset['test_filepaths'][randindexes], 1)
#         self.dispImages(dataset['test_dataset'][randindexes], 2)
        return
    def dispImages(self, images, figid):
        len_images = images.shape[0]
        rowLen = int(math.sqrt(len_images)) + 1
        colLen = int(math.sqrt(len_images)) + 1
        size = 4
        if rowLen > size:
            rowLen = size
            colLen = size
        count = 1
        plt.figure(figid)
        for row in range(rowLen):
            for col in range(colLen):
                ax=plt.subplot(rowLen, colLen, count)
                ax.imshow(images[count-1])
                count = count + 1       
        return
    def dispImages_filepath(self, images, figid):
        len_images = images.shape[0]
        rowLen = int(math.sqrt(len_images)) + 1
        colLen = int(math.sqrt(len_images)) + 1
        size = 4
        if rowLen > size:
            rowLen = size
            colLen = size
        count = 1
        plt.figure(figid)
        for row in range(rowLen):
            for col in range(colLen):
                ax=plt.subplot(rowLen, colLen, count)
                filepath = images[count-1] 
                print(filepath)
#                 filepath = filepath.replace("\\", "", 1)
                img=mpimg.imread(filepath)
                ax.imshow(img)
                count = count + 1       
        return
    def run(self):
        randindexes = np.random.permutation(1000)
#         self.checknotMNIST_2_pickle(randindexes)
        self.checkOriginalImage(randindexes)
        self.checkImageinPicle(randindexes)
        plt.show()
        return


if __name__ == "__main__":
    ci = CheckImage()
    ci.run()
    plt.show()