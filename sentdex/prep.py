#!/usr/bin/python3
import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle


class Prep():
    IMG_SIZE = 50
    DAISY = "flowers/daisy"
    DANDELION = "flowers/dandelion"
    ROSE = "flowers/rose"
    SUNFLOWER = "flowers/sunflower"
    TULIP = "flowers/tulip"
    LABELS = {DAISY: 0, DANDELION: 1, ROSE: 2, SUNFLOWER: 3, TULIP: 4}
    training_data = []

    daisy_count = 0
    dandelion_count = 0
    rose_count = 0
    sunflower_count = 0
    tulip_count = 0

    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        # do something like print(np.eye(2)[1]), just makes one_hot
                        self.training_data.append(
                            [np.array(img), np.eye(5)[self.LABELS[label]]])
                        # print(np.eye(2)[self.LABELS[label]])
                        if label == self.DAISY:
                            self.daisy_count += 1
                        elif label == self.DANDELION:
                            self.dandelion_count += 1
                        elif label == self.ROSE:
                            self.rose_count += 1
                        elif label == self.SUNFLOWER:
                            self.sunflower_count += 1
                        elif label == self.TULIP:
                            self.tulip_count += 1
                    except Exception as e:
                        pass
                        # print(label, f, str(e))
        # pickle.dump(self.training_data, open('training_data', 'wb'))
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Daisy:', self.daisy_count)
        print('Dandelion:', self.dandelion_count)
        print('Rose:', self.rose_count)
        print('Sunflower:', self.sunflower_count)
        print('Tulip:', self.tulip_count)


p = Prep()
p.make_training_data()
