from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import multi_gpu_model
import numpy as np
from PIL import Image
import argparse
import math
from setup import *
from generator import *


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(64*90*90))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((90, 90, 64), input_shape=(64*90*90,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(
        Conv2D(64, (5, 5),
               padding='same',
               input_shape=(360, 360, 3))
    )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    '''
    Put all the generated images in a matrix
    '''
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(batch_size, dataset_directory, epochs, target_size=(360, 360)):
    multi_gpu = False # hardcoded 
    train_generator = Generator(dataset_directory, batch_size, target_size)
    total_train_data = train_generator.get_len_total_data()
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = (X_train.astype(np.float32) - 127.5)/127.5
    # X_train = X_train[:, :, :, None]
    # print(X_train.shape)
    # X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()
    print("Making d and g multi gpu")
    if multi_gpu:
        d = multi_gpu_model(d, gpus=8)
        g = multi_gpu_model(g, gpus=8)
    d_on_g = generator_containing_discriminator(g, d)
    print("making multi gpu")
    if multi_gpu:
        d_on_g = multi_gpu_model(d_on_g, gpus=8)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(total_train_data/batch_size))
        total_steps = math.ceil(total_train_data / batch_size)
        step = 0
        for X_train in train_generator.generate():
            step += 1
            if step == total_steps:
                break

            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            # image_batch = X_train[index*batch_size:(index+1)*batch_size]
            generated_images = g.predict(noise, verbose=0)
            print(f'Generated images from generator {generated_images.shape}')
            if step % 20 == 0:
                # to save images
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(step)+".png")
            X = np.concatenate((X_train, generated_images))
            y = [1] * batch_size + [0] * batch_size
            print('Starting training discriminator')
            d_loss = d.train_on_batch(X, y)
            print('Done')
            print("batch %d d_loss : %f" % (step, d_loss))
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            d.trainable = False
            # to train generator, discriminator weights are frozen
            print('Started training generator')
            g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
            print('Done')
            d.trainable = True
            print("batch %d g_loss : %f" % (step, g_loss))
            if step % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)


def generate(batch_size, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (batch_size*20, 100))
        generated_images = g.predict(noise, verbose=1)
        print(f'Generted images shape is {generated_images.shape}')
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size*20)
        index.resize((batch_size*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros(
            (batch_size,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (batch_size, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    dataset_directory = f'../dataset2/train/'
    batch_size = 4
    epochs = 100
    gpu_id = '5'
    # setup_gpu(gpu_id)
    args = get_args()

    if args.mode == "train":
        train(args.batch_size, dataset_directory, epochs)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, nice=args.nice)
