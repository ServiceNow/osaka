# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# These Omniglot loaders are from Jackie Loong's PyTorch MAML implementation:
#     https://github.com/dragen1860/MAML-Pytorch
#     https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot.py
#     https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglotNShot.py

import  torchvision.transforms as transforms
from    PIL import Image
import  numpy as np

import torch
import  torch.utils.data as data
import  os
import  os.path
import  errno
from pdb import set_trace


class Omniglot_(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None,
                 download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour

def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


class Omniglot:

    def __init__(self, root):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = 28
        self.device = 'cpu'
        if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
            # if root/data.npy does not exist, just download it
            self.x = Omniglot_(
                root, download=True,
                transform=transforms.Compose(
                    [lambda x: Image.open(x).convert('L'),
                     lambda x: x.resize((28, 28)),
                     lambda x: np.reshape(x, (28, 28, 1)),
                     lambda x: np.transpose(x, [2, 0, 1]),
                     lambda x: x/255.]),
            )

            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in self.x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.data = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.data.append(np.array(imgs))

            # as different class may have different number of imgs
            self.data = np.array(self.data).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
            # each character contains 20 imgs
            print('data shape:', self.data.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, 'omniglot.npy'), self.data)
            print('write into omniglot.npy.')
        else:
            # if data.npy exists, just load it.
            self.data = np.load(os.path.join(root, 'omniglot.npy'))
            print('load from omniglot.npy.')





