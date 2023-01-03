import numpy as np
import matplotlib.pyplot as plt
import os 

class Dataset:
    def __init__(self, path):
        # read bmp file
        img_paths = os.listdir(path)
        # print(img_paths)
        # sort the file name
        img_paths.sort()
        print(img_paths)
        for img_path in img_paths:
            if not img_path.endswith(".bmp"):
                continue
            img = plt.imread(path + img_path)
            self.u, self.v = img.shape[0], img.shape[1]
            img = img.reshape(1, self.u, self.v, -1)
            if img_path == img_paths[0]:
                self.data = img
            else:
                self.data = np.concatenate((self.data, img), axis=0)
        self.num = self.data.shape[0]
        self.s = np.sqrt(self.data.shape[0]).astype(int)
        self.t = np.sqrt(self.data.shape[0]).astype(int)
        assert self.s * self.t == self.num
        self.camera_shape = [self.s, self.t]
        self.image_shape  = [self.u, self.v]
        self.data = self.data.reshape(self.camera_shape + self.image_shape + [-1])
        print("Read {self.num:d} images, each image is {self.u:d} x {self.v:d}.".format(self=self))

    def show(self, idx):
        plt.imshow(self.data[idx, :, :, :])
        plt.show()

    def show_all(self):
        fig = plt.figure()
        for i in range(self.s):
            for j in range(self.t):
                sub = fig.add_subplot(1, self.t, j+1)
                sub.imshow(self.data[i, j])
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
            plt.show()

if "__main__" == __name__:
    pass
    # path = '../data/'
    # data = Dataset(path)
    # data.show_all()