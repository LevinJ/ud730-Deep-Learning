
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from A1_notmnistdataset import p2_checkimage

class DataExploration:
    def __init__(self):
        with open('notMNIST_2.pickle', 'rb') as f:
            self.dataset = pickle.load(f)
        self.train_dataset = self.dataset['train_dataset']
        self.train_labels = self.dataset['train_labels']
        
        self.valid_dataset = self.dataset['valid_dataset']
        self.valid_labels = self.dataset['valid_labels']
        
        self.test_dataset = self.dataset['test_dataset']
        self.test_labels = self.dataset['test_labels']
        
        self.train_filepaths = self.dataset['train_filepaths']
        self.test_filepaths = self.dataset['test_filepaths']
        self.valid_filepaths = self.dataset['valid_filepaths']
        
#         print('Training:', self.train_dataset.shape, self.train_labels.shape)
#         print('Validation:', self.valid_dataset.shape, self.valid_labels.shape)
#         print('Testing:', self.test_dataset.shape, self.test_labels.shape)
        return
    def checkDupwithin(self, images, groupname, image_paths=None):
        temp_dict = {}
        dup_dict={}
        images.flags.writeable = False
        count = 0
        for idx, item in enumerate(images):
            if item.data in temp_dict:
                count = count + 1
#                 print("duplicate {}:{}".format(count, idx))
                existingId = temp_dict[item.data]
                if not (existingId in dup_dict):
                    dup_dict[existingId] = []
                dup_dict[existingId].append(idx)  
                continue
            temp_dict[item.data] = idx
        print("{} has {} duplicate items, {} total items".format(groupname, count, images.shape[0])) 
#         print(dup_dict) images[dup_dict[25]]
#         di = dup_dict[128]
#         di = dup_dict[1018]
#         self.dispImages(image_paths[di], images[di])
        return
    def checkDupBetween(self,datasetA, datasetB,lablesB, A,B, Apaths = None, Bpaths=None):
        temp_dict = {}
        dup_dict={}
        datasetA.flags.writeable = False
        datasetB.flags.writeable = False
        count = 0
        #build up base table for datasetA in temp_dict
        for idx, item in enumerate(datasetA):
            if item.data in temp_dict: 
                continue
            temp_dict[item.data] = idx
        for idx,img in enumerate(datasetB):
            if img.data in temp_dict:
                count = count + 1
                existingId = temp_dict[img.data] 
                if not (existingId in dup_dict):
                        dup_dict[existingId] = []
                dup_dict[existingId].append(idx)  
                
        print("{} {} duplicate {}, total count {}, total count {}".format(A, B, count, datasetA.shape[0], datasetB.shape[0]))
#         print(Apaths[16812])
#         print(Bpaths[dup_dict[16812]])
          
        return
    def getDiffLableCount(self, dup_table, labelsB):
        count = 0
        for key, value in dup_table.iteritems(): 
            truelabels = [labelsB[item]  for item in value]
            allthesame = all(truelabels[0] == item for item in truelabels)
            if not allthesame:
                count = count + 1
                print(truelabels)
        return count
    def dispImages(self, filepaths, images):
        ci = p2_checkimage.CheckImage()     
        ci.dispImages_filepath(filepaths, 1) 
        ci.dispImages(images, 2)
        return
    def run(self):

        
        self.checkDupwithin(self.train_dataset,'train_dataset', self.train_filepaths)
        self.checkDupwithin(self.test_dataset,'test_dataset')
        self.checkDupwithin(self.valid_dataset,'validation_dataset')
        self.checkDupBetween(self.train_dataset, self.test_dataset,self.test_labels,'train_dataset','test_dataset', self.train_filepaths, self.test_filepaths)
        self.checkDupBetween(self.train_dataset, self.valid_dataset,self.test_labels,'train_dataset','validation_dataset')
        self.checkDupBetween(self.valid_dataset, self.test_dataset,self.test_labels,'validation_dataset','test_dataset')
        plt.show()
        return
    


if __name__ == "__main__":   
    obj= DataExploration()
    obj.run()
