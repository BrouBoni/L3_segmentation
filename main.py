import os

from dataloader import Dataset
from model import Model

if __name__ == '__main__':

    path = '/home/brouboni/PycharmProjects/L3_segmentation/dataset_new/'
    dataset = Dataset(path)

    checkpoints_dir = 'checkpoints/'
    name = 'exp'

    expr_dir = os.path.join(checkpoints_dir, name)

    model = Model(expr_dir, n_blocks=9, niter=150, niter_decay=150)
    training_loader_patches, validation_loader_patches = dataset.get_loaders()
    model.train(training_loader_patches, validation_loader_patches)
