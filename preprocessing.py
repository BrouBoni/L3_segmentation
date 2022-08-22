import os

import torch
import torchio as tio
from PIL import ImageOps
from glob2 import glob
from natsort import natsorted


# Function to convert RGB to YCbCr
def rgb2label(path):
    image = tio.ScalarImage(path).data
    r = image[0, :, :, :]
    g = image[1, :, :, :]
    b = image[2, :, :, :]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (-0.1687 * r - 0.3313 * g + 0.5 * b) + 128
    cr = (0.5 * r - 0.419 * g - 0.081 * b) + 128

    image = torch.stack((y, cb, cr), dim=0)
    image = image[1, :, :, :]
    image = torch.less(image, 128).int()

    return image[None, :]


if __name__ == '__main__':

    path = "/home/brouboni/PycharmProjects/L3_segmentation/dataset_new/"

    raw_files = natsorted(glob(os.path.join(path, 'data*')))
    m_files = natsorted(glob(os.path.join(path, 'M_*')))
    efat_files = natsorted(glob(os.path.join(path, 'EFAT_*')))
    int_files = natsorted(glob(os.path.join(path, 'INT_*')))
    ifat_files = natsorted(glob(os.path.join(path, 'IFAT_*')))

    for index, raw_file in enumerate(raw_files):
        ct = tio.ScalarImage(raw_file)
        label_map = torch.zeros(ct.shape, dtype=torch.long)

        m_label = rgb2label(m_files[index])
        efat_label = rgb2label(efat_files[index])
        int_label = rgb2label(int_files[index])
        ifat_label = rgb2label(ifat_files[index])

        in_label = int_label - ifat_label

        label_map += efat_label + 2 * m_label + 3 * ifat_label + 4 * in_label
        label_map[label_map > 4] = 0
        label = tio.LabelMap(tensor=label_map, affine=ct.affine)

        label_grey = ImageOps.grayscale(label.as_pil())
        label_grey.save(os.path.join(path, 'label_sbj_'+str(index+1)+'.png'))
