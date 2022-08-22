import os

import numpy as np


def print_log(out_f, message):
    out_f.write(message + "\n")
    out_f.flush()
    print(message)


def format_train_log(epoch, iteration, errors, t):
    message = '(epoch: %d, iteration: %d, time: %.3f) ' % (epoch, iteration, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def format_validation_log(epoch, iteration, errors, t):
    message = '(epoch: %d, iteration: %d, time: %.3f) ' % (epoch, iteration, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def listdir_full_path(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def visualize_training(visuals):
    ct = visuals['ct'].cpu().squeeze()
    segmentation_mask = visuals['segmentation_mask'].cpu().argmax(dim=1, keepdim=True).squeeze()
    fake_segmentation_mask = visuals['fake_segmentation_mask'].cpu().argmax(dim=1, keepdim=True).squeeze()
    return ct, segmentation_mask, fake_segmentation_mask


def decode_segmentation(image, nc=5):
    label_colors = np.array([(0, 0, 0),
                             # 0=background, 1=efat, 2=m, 3=ifat, 4=in
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for line in range(0, nc):
        idx = image == line
        r[idx] = label_colors[line, 0]
        g[idx] = label_colors[line, 1]
        b[idx] = label_colors[line, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb
