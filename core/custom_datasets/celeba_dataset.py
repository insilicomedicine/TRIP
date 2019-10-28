# code based on
# https://github.com/rasbt/deep-learning-book/blob/master/code/model_zoo/pytorch_ipynb/custom-data-loader-celeba.ipynb
# and
# https://gist.github.com/charlesreid1/4f3d676b33b95fce83af08e4ec261822#file-get_drive_file-py

import math
from tqdm import tqdm

import numpy as np
import os

from torch.utils.data import Dataset
from PIL import Image

import requests

import zipfile

import torch

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, data_dir, download=False, transform=None, seed=777,
                 train=True):
        os.makedirs(data_dir, exist_ok=True)

        celeba_dir = os.path.join(data_dir, 'celeba')
        zip_file = os.path.join(data_dir, 'celeba.zip')
        self.img_dir = os.path.join(data_dir, 'celeba/img_align_celeba/')

        if download and (not os.path.isdir(celeba_dir)):
            if (not os.path.isfile(zip_file)):
                print('downloading ' + zip_file)
                self.__download_file_from_google_drive__(
                    '0B7EVK8r0v71pZjFTYXZWM3FlRnM', zip_file)

        if not os.path.isdir(celeba_dir):
            print('unziping ' + zip_file + ' into ' + celeba_dir)
            zip_ref = zipfile.ZipFile(zip_file, 'r')
            zip_ref.extractall(celeba_dir)
            zip_ref.close()
            print('deleting ' + zip_file)
            os.remove(zip_file)

        np.random.seed(seed)

        all_img_names = os.listdir(self.img_dir)
        train_test_perm = np.random.permutation(len(all_img_names))

        if train:
            self.img_names = [all_img_names[i] for i in
                              train_test_perm[:round(0.9 * len(all_img_names))]]
        else:
            self.img_names = [all_img_names[i] for i in
                              train_test_perm[round(0.9 * len(all_img_names)):]]

        self.img_names = np.array(self.img_names)

        self.transform = transform

    @staticmethod
    def __download_file_from_google_drive__(id, destination):
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 1024 * 1024

            with open(destination, "wb") as f:
                for chunk in tqdm(response.iter_content(CHUNK_SIZE),
                                  total=math.ceil(1443490838 // CHUNK_SIZE),
                                  unit='MB', unit_scale=True):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.ones(1)

    def __len__(self):
        return self.img_names.shape[0]