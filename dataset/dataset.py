from torch.utils.data import Dataset
from pathlib import Path
import cv2
from PIL import Image
import numpy as np



class VascularDataset(Dataset):
    """
    Dataset class for the vascular dataset:

    """
    def __init__(self,
                 images_list,
                 transform=None):
        super(VascularDataset, self).__init__()
        self.images = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = str(self.images[index])
        seg_path = img_path.replace('images', 'masks')
        img = cv2.imread(img_path)

        seg = Image.open(seg_path)
        seg = seg.convert('RGB')    # remove the transparent portion of the image
        seg = seg.convert('L')      # from RGB to black and white
        seg = seg.point(lambda x: 0 if x < 1 else 255.0, '1')
        seg = np.array(seg)  # equivalent to a cv2 image

        if self.transform:
            transformed = self.transform(image=img, mask=seg)
            img = transformed['image']
            seg = transformed['mask']
        return img, seg


def gen_split(root_dir, valid_ids):
    """

    :param root_dir:
    :param valid_ids:
    :return:
    """
    train_dataset = []
    valid_dataset = []
    # extract all the folder related to patients, i.e. root_dir/Train/cancer_type/patient
    patients = sorted(Path(root_dir).glob("Train/images/*"))
    for patient in patients:
        if str(patient).split("/")[-1] in valid_ids:
            valid_dataset += sorted(Path(patient).glob('*'))
        else:
            train_dataset += sorted(Path(patient).glob('*'))
    return train_dataset, valid_dataset


def generate_datasets(data_dir, valid_ids=None):
    """
    Function that automatically generates training and validation sets for the training phase

    :param data_dir: path to the data directory generated with makeDataset.py
    :param valid_ids: list of the patients to use as validation set
    :return:
        if valid_ids is not None:
            train_set: VascularDataset containing the training data split
            valid_set: VascularDataset containing the validation data split
        otherwise:
            train_set: VascularDataset containing the training data split
    """
    if valid_ids is None:
        valid_ids = []
    train_list, val_list = gen_split(data_dir, valid_ids)
    if len(valid_ids) == 0:
        return VascularDataset(train_list)
    else:
        return (VascularDataset(train_list),    # Training Set
                VascularDataset(val_list))      # Validation Set


