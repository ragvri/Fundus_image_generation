import os
import sys
import pandas as pd
import subprocess


train_images = os.listdir('./dataset2/train/')

test_images = os.listdir('./dataset2/test/')
train_images = [k for k in train_images if 'jpeg' in k]
test_images = [k for k in test_images if 'jpeg' in k]
merged = train_images + test_images


# this contains the final set of images

# * now we read the csv to get the details of the labels

df = pd.read_csv('./dataset2/trainLabels.csv')

index4 = df.index[df['level'] == 4].tolist()
index3 = df.index[df['level'] == 3].tolist()
index2 = df.index[df['level'] == 2].tolist()
index1 = df.index[df['level'] == 1].tolist()
index0 = df.index[df['level'] == 0].tolist()

images4 = df.iloc[index4]['image'].tolist()
images3 = df.iloc[index3]['image'].tolist()
images2 = df.iloc[index2]['image'].tolist()
images1 = df.iloc[index1]['image'].tolist()
images0 = df.iloc[index0]['image'].tolist()


# for img in images4:
#     cmd = "cp ./dataset2/train/{}.jpeg ./train4/{}.jpeg".format(img, img)
#     subprocess.call(cmd.split(' '))


# for img in images0:
#     cmd = "cp ./dataset2/train/{}.jpeg ./train0/{}.jpeg".format(img, img)
#     subprocess.call(cmd.split(' '))
    
for img in images1:
    cmd = "cp ./dataset2/train/{}.jpeg ./train1/{}.jpeg".format(img, img)
    subprocess.call(cmd.split(' '))
for img in images2:
    cmd = "cp ./dataset2/train/{}.jpeg ./train2/{}.jpeg".format(img, img)
    subprocess.call(cmd.split(' '))
for img in images3:
    cmd = "cp ./dataset2/train/{}.jpeg ./train3/{}.jpeg".format(img, img)
    subprocess.call(cmd.split(' '))