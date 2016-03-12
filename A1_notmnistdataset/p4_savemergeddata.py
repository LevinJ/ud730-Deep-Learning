import pickle
import os
from A1_notmnistdataset import p3_mergedata
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pickle_file = 'notMNIST_3.pickle'


def run(valid_dataset, valid_labels, train_dataset, train_labels,test_dataset, test_labels, train_filepaths, test_filepaths, valid_filepaths):
    print("save file ", pickle_file)
    try:
        f = open(pickle_file, 'wb')
        save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
                'train_filepaths':train_filepaths,
                'test_filepaths': test_filepaths,
                'valid_filepaths': valid_filepaths
                }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
            


class SaveMergedData:
    def checkImageinPicle(self):
        plt.figure(1)
        with open(pickle_file, 'rb') as handle:
            dataset = pickle.load(handle)
        plt.imshow(dataset['train_dataset'][100])
        plt.show()
        return
    def run(self):
        if os.path.exists(pickle_file):
            print("file ", pickle_file, " already exist")
            statinfo = os.stat(pickle_file)
            print('Compressed pickle size:', statinfo.st_size)
            return
        obj= p3_mergedata.MergeData()
        res = obj.merge()
        valid_dataset, valid_labels, train_dataset, train_labels,test_dataset, test_labels, train_filepaths, test_filepaths, valid_filepaths = res
        run(valid_dataset, valid_labels, train_dataset, train_labels,test_dataset, test_labels, train_filepaths, test_filepaths, valid_filepaths)
        return
if __name__ == "__main__":   
    obj= SaveMergedData()
    obj.run()
#     obj.checkImageinPicle()