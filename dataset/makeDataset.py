from pathlib import Path
import os
import cv2
import argparse
import albumentations.augmentations.functional as F


INPUT_DIM = 668
IMG_DIM = 2000 + 134 * 2
SLICE_LEN = 400
N = 5
PADDING = 134


def crop_images(img_list, seg_list, imag_path, seg_path):
    """
    Function that crops the image into N = 2000 / SLICE_LEN tiles

    :param img_list: list of the images to crop
    :param seg_list: list of the segmentations to crop
    :param imag_path: path where to save the cropped images
    :param seg_path: path where to save the cropped segmentations
    :return:
    """
    for idx, (img_path, mask_path) in enumerate(zip(img_list, seg_list)):
        image = cv2.imread(str(img_path))
        image = F.pad(image, min_height=IMG_DIM, min_width=IMG_DIM)

        mask = cv2.imread(str(mask_path))
        mask = F.pad(mask, min_height=IMG_DIM, min_width=IMG_DIM)

        tile = 0
        for i in range(0, 2000, SLICE_LEN):
            for j in range(0, 2000, SLICE_LEN):
                tile += 1
                cv2.imwrite(os.path.join(imag_path, f"{idx:04}_{tile:02}.png"),
                            image[i:i + INPUT_DIM, j:j + INPUT_DIM])
                cv2.imwrite(os.path.join(seg_path, f"{idx:04}_{tile:02}.png"),
                            mask[i:i + INPUT_DIM, j:j + INPUT_DIM])


def make_dataset(in_path, out_path):
    """
    Take the original dataset and creates the cropped version of the original dataset

    :param in_path: path of the original dataset
    :param out_path: path for the cropped dataset
    :return:
    """
    # Training Set
    train_mask_path = os.path.join(out_path, "Train/masks")
    train_imag_path = os.path.join(out_path, "Train/images")

    Path(train_mask_path).mkdir(parents=True, exist_ok=True)
    Path(train_imag_path).mkdir(parents=True, exist_ok=True)

    patients = sorted(Path(in_path).glob("Train/*/*/"))
    dataset_descr = "patient_id, patient, cancer_type, num_imgs\n"
    for patient_id, patient in enumerate(patients):
        imgs = sorted(Path(patient).glob("*/[!*seg*]*.png"))
        segs = sorted(Path(patient).glob("*/*seg*.png"))
        print(str(patient).split("/")[-2], "\t", str(patient).split("/")[-1])

        dataset_descr += f'{patient_id}, {str(patient).split("/")[-1]}, {str(patient).split("/")[-2]}, {str(len(imgs))}\n'
        crop_images(imgs, segs,
                    os.path.join(train_imag_path, f"{patient_id}"),
                    os.path.join(train_mask_path, f"{patient_id}"))
    print(dataset_descr)
    with open(os.path.join(out_path, "train_descr.csv"), "w+") as f:
        f.write(dataset_descr)

    # Test Set
    test_mask_path = os.path.join(out_path, "Test/masks")
    test_imag_path = os.path.join(out_path, "Test/images")

    Path(test_mask_path).mkdir(parents=True, exist_ok=True)
    Path(test_imag_path).mkdir(parents=True, exist_ok=True)

    cancer_list = sorted(Path(in_path).glob("Test/*/"))
    dataset_descr = "patient_id, patient, cancer_type, num_imgs\n"
    for cancer_id, cancer in enumerate(cancer_list):
        imgs = sorted(Path(cancer).glob("[!*seg*]*.png"))
        segs = sorted(Path(cancer).glob("*seg*.png"))
        dataset_descr += f'{cancer_id}, ,{str(cancer).split("/")[-2]}, {str(len(imgs))}\n'
        crop_images(imgs, segs,
                    os.path.join(test_imag_path, f"{cancer_id}"),
                    os.path.join(test_mask_path, f"{cancer_id}"))
    print(dataset_descr)
    with open(os.path.join(out_path, "train_descr.csv"), "w+") as f:
        f.write(dataset_descr)


# in_path = './drive/MyDrive/Bioinformatics/vascular_segmentation/'
# out_path = './drive/MyDrive/Bioinformatics/dataset'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize the vascular_segmentation\
    dataset into image/mask directories')
    parser.add_argument('in_path', type=str, help='input path')
    parser.add_argument('out_path', type=str, help='output path')

    args = parser.parse_args()
    make_dataset(args.in_path, args.out_path)
