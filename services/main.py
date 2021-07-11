from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from PIL import Image
from tkinter import Tk, Button, Listbox
import cv2
import time
from Crypto.Cipher import AES
from Crypto import Random
import csv

workers = 0 if os.name == 'nt' else 4

keys_df = pd.read_csv('../try/data/keys_list.csv')
# print(keys_df,'\n')

def destroy_images(check=False):
    rootdir = '../try/data/test_images'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if (subdir[-4:] == 'temp') and (check == True):
                if file.endswith('.jpg') or file.endswith('.png'):
                    os.remove(os.path.join(subdir,file))
            elif (subdir[-4:] != 'temp') and (file.endswith('.jpg') or file.endswith('.png')):
                os.remove(os.path.join(subdir,file))


def decrypt_images():
    rootdir = '../try/data/test_images'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filedir = os.path.join(subdir, file)[7:].replace('\\','/')
            if filedir in keys_df['enc_img_path'].tolist():
                timestr = time.strftime("%Y%m%d-%H%M%S")
                input_file = open(os.path.join(subdir, file), 'rb')
                input_data = input_file.read()
                input_file.close()

                row = keys_df.loc[keys_df['enc_img_path'] == filedir]

                cfb_cipher = AES.new(bytes.fromhex(row['key'].iloc[0]), AES.MODE_CFB,bytes.fromhex(row['iv'].iloc[0]))
                enc_data = cfb_cipher.decrypt(input_data)

                dec_file = open("{}".format(os.path.join(subdir,file).replace('\\','/').replace('.enc','.png')), "wb")
                dec_file.write(enc_data)
                dec_file.close()

def mtcnn_init(): #function to construct the face detection module
    #determine if nvidia GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    #Create the MTCNN module
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    return mtcnn, device

def distance_matrix(device): #face recognition using similarity/distance of images.

    #define Inception Resnet V1 module
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    #define a dataset and data loader
    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder(os.path.join('data/test_images'))
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    #perform mtcnn facial detection
    aligned = []
    names = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            # print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    #calculate image embeddings
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    # print(embeddings)

    #determine distance matrix for classes
    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    df = pd.DataFrame(dists, columns=names, index=names)

    #find possible labels to input image
    possible_labels = df.query('temp < 1')[:-1].index.tolist()
    possible_labels = np.unique(np.array(possible_labels))

    #destroy decrypted images
    destroy_images()

    print('Choose the appropriate option: ')
    for i in range(len(possible_labels)):
        print(i+1,possible_labels[i])

    print(len(possible_labels)+1,'None of the above')
    option = input('Enter the appropriate option: ')

    if option == len(possible_labels)+1:
        print('Prompt user to their directory')
        # return something

    else:
        return possible_labels[int(option)-1]

def capture_cam(key,iv): #function to capture, store and encrypt images from camera
    cap = cv2.VideoCapture(0)
    img_name = 0

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        k = cv2.waitKey(1)
        if k%256 == 32: # SPACE pressed

            #create a unique filename for the image and capture it
            timestr = time.strftime("%Y%m%d-%H%M%S")
            img_name = "data/test_images/temp/img_{}.png".format(timestr)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))

            #call distance matrix function
            label = distance_matrix(device)

            #encrypt the stored image
            input_file = open(img_name, 'rb')
            input_data = input_file.read()
            input_file.close()

            cfb_cipher = AES.new(key, AES.MODE_CFB, iv)
            enc_data = cfb_cipher.encrypt(input_data)

            enc_file = open("data/test_images/{}/img_{}_encrypted.enc".format(label,timestr), "wb")
            enc_file.write(enc_data)
            enc_file.close()

            #store key and iv for encrypted image in a csv
            new_data = ["data/test_images/{}/img_{}_encrypted.enc".format(label,timestr),key.hex(),iv.hex()]
            with open('../try/data/keys_list.csv','a') as f:
                writer = csv.writer(f)
                writer.writerow(new_data)
            break

    cap.release()
    cv2.destroyAllWindows()

    # return cv2.imread(img_name)

def plot(): #driver function
    #decrypt image database
    # decrypt_images()
    #
    # #initialize key and initial vector for AES encryption
    # key = Random.new().read(AES.block_size)
    # iv = Random.new().read(AES.block_size)

    #call and store webcam image into a variable
    capture_cam(key,iv)

    #close application
    # root.destroy()

    #destroy temp images
    destroy_images(True)

# root = Tk()
# mtcnn, device = mtcnn_init()
#
# root.title('Vaccination Detector')
#
# B = Button(root, text ="Take Photo", command = plot)
# B.pack()
# # # dimensions of the main window
# root.geometry("500x500")
#
# root.mainloop()
