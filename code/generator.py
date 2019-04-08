import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class Generator:
    def __init__(self, dataset_directory, batch_size, target_size):
        self.dataset_directory = dataset_directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.i = 0
        self.image_list = os.listdir(self.dataset_directory)

    def get_image(self, image_file_name, augment=True):

        img_path = f'{self.dataset_directory}/{image_file_name}'
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.target_size)
        k = np.expand_dims(img, 0)
        img = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2,
                                 zoom_range=0.3, horizontal_flip=True, fill_mode='nearest').flow(k).next()
        return img

    def generate(self):
        total_data = self.get_len_total_data()
        print(f'Total data: {total_data}')
        while True:

            batch_images = np.zeros((self.batch_size, self.target_size[0], self.target_size[1], 3))
            
            count = 0
            while count<self.batch_size:
                img = self.get_image(self.image_list[self.i])
                batch_images[count] = img
                count+=1
                self.i+=1
                self.i = self.i%total_data
                
            # yield batch_images
            yield batch_images

    def get_len_total_data(self):
        return len(os.listdir(self.dataset_directory))

if __name__ == "__main__":
    dataset_directory = f'../dataset2/train/'
    a = Generator(dataset_directory, 4, (360,360))
    print(dataset_directory)
    print("OO")
    for x in a.generate():
        print(x.shape)