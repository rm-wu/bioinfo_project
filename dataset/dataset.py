from torch.utils.data import Dataset
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class VascularDataset(Dataset):
    """
    Dataset class for the vascular dataset:

    """
    def __init__(self,
                 images_list,
                 transform=None,
                 load_in_memory=True):
        super(VascularDataset, self).__init__()
        self.images = images_list
        self.mean_normalization=(0.485, 0.456, 0.406)
        self.std_normalization=(0.229, 0.224, 0.225)
        self.image_transform = A.Compose(
        [
            A.Normalize(mean=self.mean_normalization, std=self.mean_normalization),
            ToTensorV2()
        ])
        self.transform=transform
        self.ToTensor=A.Compose([ToTensorV2()])
        self.load_in_memory = load_in_memory
        if load_in_memory:
            self._map = dict()

    def __len__(self):
        return len(self.images)

    def set_normalization(self, mean, std):
        self.mean_normalization=mean
        self.std_normalization=std

    def compute_normalization(self):
        rmean = 0
        rstd = 0
        gmean = 0
        gstd = 0
        bmean = 0
        bstd = 0
        count =0
        for img_path in self.images:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
            image = img[134:534, 134:534, :]
            r = np.reshape(image[:, :, 0], -1)
            g = np.reshape(image[:, :, 1], -1)
            b = np.reshape(image[:, :, 2], -1)
            rmean += r.mean()
            rstd += r.std()
            gmean += g.mean()
            gstd += g.std()
            bmean += b.mean()
            bstd += b.std()
            count += 1
        return (rmean,gmean,bmean), (rstd, gstd, bstd)

    def __getitem__(self, index):
        if self.load_in_memory and \
                index in self._map:
            img, seg = self._map[index]
        else:
            img_path = str(self.images[index])
            seg_path = img_path.replace('images', 'masks')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # TODO: alternativa for binarize the segmentation mask
            seg = Image.open(seg_path)
            seg = seg.convert('RGB')    # remove the transparent portion of the image
            seg = seg.convert('L')      # from RGB to black and white
            seg = seg.point(lambda x: 0 if x < 1 else 255.0, '1')
            seg = np.array(seg, dtype=np.float32)  # equivalent to a cv2 image
            seg = np.expand_dims(seg, axis=0)

            if self.load_in_memory:
                self._map[index] = (img, seg)

        if self.transform:
            transformed = self.transform(image=img, mask=seg)
            img = transformed['image']
            seg = transformed['mask']

        img = self.image_transform(image=img)['image']
        return img, seg


def gen_split(root_dir, valid_ids, train_or_test='Train'):
    """

    :param root_dir:
    :param valid_ids:
    :return:
    """
    if train_or_test=='Train':
        train_dataset = []
        valid_dataset = []
        # extract all the folder related to patients, i.e. root_dir/Train/cancer_type/patient
        train_images = "Train" + os.sep + "images" + os.sep + "[!.]*"
        patients = sorted(Path(root_dir).glob(train_images))

        for patient in patients:
            if str(patient).split(os.sep)[-1] in valid_ids:
                valid_dataset += sorted(Path(patient).glob('*'))
            else:
                train_dataset += sorted(Path(patient).glob('*'))
        return train_dataset, valid_dataset

    elif train_or_test=='Test':
        test_dataset = []
        # extract all the folder related to patients, i.e. root_dir/Train/cancer_type/patient
        test_images = "Test" + os.sep + "images" + os.sep + "[!.]*"
        patients = sorted(Path(root_dir).glob(test_images))
        for patient in patients:
            test_dataset+=sorted(Path(patient).glob('*'))
        return test_dataset

def generate_datasets(data_dir, train_or_test, valid_ids=None, load_in_memory=True, train_transform=None):
    """
    Function that automatically generates training and validation sets for the training phase

    :param data_dir: path to the data directory generated with makeDataset.py
    :param valid_ids: list of the patients to use as validation set
    :param load_in_memory: load the images in RAM
    :return:
        if valid_ids is not None:
            train_set: VascularDataset containing the training data split
            valid_set: VascularDataset containing the validation data split
        otherwise:
            train_set: VascularDataset containing the training data split
    """
    if train_or_test=='Train':
        if valid_ids is None:
            valid_ids = []
        train_list, val_list = gen_split(data_dir, valid_ids, train_or_test)
        if len(valid_ids) == 0:
            return VascularDataset(train_list, transform=train_transform)
        else:
            return (VascularDataset(train_list,
                                    load_in_memory=load_in_memory, transform=train_transform),    # Training Set
                    VascularDataset(val_list,
                                    load_in_memory=load_in_memory))      # Validation Set
    elif train_or_test=='Test':
        test_list = gen_split(data_dir, valid_ids, train_or_test)
        return VascularDataset(test_list)


if __name__ == "__main__":
    data_dir = './data'
    train_dataset, val_dataset = generate_datasets(data_dir, valid_ids=['0', '1'])
