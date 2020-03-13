from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import pickle as pkl
#import cPickle as pkl
from io import BytesIO
from torchvision import transforms as t
import numpy as np
from pdb import set_trace

class NonEpisodicTieredImagenet(Dataset):
    tasks_type = "clss"
    name = "tiered-imagenet"
    split_paths = {"train":"train", "test":"test", "valid": "val", "val": "val"}
    c = 3
    h = 64
    w = 64

    def __init__(self, path, split, transforms=t.ToTensor(), **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        split = self.split_paths[split]
        self.ROOT_PATH = path
        if not os.path.exists(self.ROOT_PATH+'/{}-tiered-imagenet.npy'.format(split)): #'/tmp/{}-tiered-imagenet.pkl'.format(split)):
            #TODO(next line is missing something)
            if not os.path.exists(self.ROOT_PATH+'/{}-tiered-imagenet.npy') :
                raise Exception("Please download tiered-imagenet as indicated"
                        "in https://github.com/renmengye/few-shot-ssl-public")
            img_path = os.path.join(self.ROOT_PATH, "%s_images_png.pkl" % (split))
            label_path = os.path.join(self.ROOT_PATH, "%s_labels.pkl" % (split))
            self.transforms = transforms
            with open(img_path, 'rb') as infile:
                images = pkl.load(infile, encoding="bytes")

            with open(label_path, 'rb') as infile:
                self.labels = pkl.load(infile, encoding="bytes")
                self.labels_specific = self.labels["label_specific"]
                self.labels_general = self.labels["label_general"]

            print("Loading tiered-imagenet...")
            label_count = {i: (self.labels_specific == i).astype(int).sum() for i in set(self.labels_specific)}
            min_count = np.min(list(label_count.values()))
            # was 84, 84
            self.data = torch.zeros(len(label_count.keys()), min_count, self.c, self.h, self.w, dtype=torch.uint8)
            label_count = {i: 0 for i in set(self.labels_specific)}
            for im, label in zip(images, self.labels_specific):
                index = label_count[label]
                if index == min_count:
                    continue
                else:
                    self.data[label, index, ...] = torch.from_numpy(np.transpose(self.__decode(im).resize((self.h, self.w),Image.NEAREST), [2,0,1]))
                    label_count[label] += 1
            np.save(os.path.join(self.ROOT_PATH,'%s-tiered-imagenet' % (split)), self.data.data.numpy(),allow_pickle=True)
            del (images)

        else:
            #with open(self.ROOT_PATH+'/{}-tiered-imagenet.pkl'.format(split), 'rb') as infile: #'/tmp/{}-tiered-imagenet.pkl'.format(split), 'rb') as infile:
            #    self.data = pkl.load(infile)
            self.data = torch.from_numpy(np.load(os.path.join(self.ROOT_PATH, '{}-tiered-imagenet.npy'.format(split)),
                                allow_pickle=True))
            #self.data = np.load(os.path.join(self.ROOT_PATH, '/{}-tiered-imagenet.pkl'.format(split)),
            #                    allow_pickle=True)
            print(self.data.size())
        print("Done")

    def from_hierarchy(self, start, end):
        ret = []
        mask = (self.labels_general >= start) * (self.labels_general < end)
        specific_set = np.unique(self.labels_specific[mask])
        return self.data[:, specific_set, ...]

    def __decode(self, image):
        return Image.open(BytesIO(image)).convert("RGB")

    def __getitem__(self, item):
        return self.transforms(self.__decode(self.images[item])), self.labels[item]

    def __len__(self):
        return self.data.shape[0] * self.data.shape[1]
