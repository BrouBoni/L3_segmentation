import os

from dataloader import Dataset
from model import Model

if __name__ == '__main__':

    path = '/home/nicoc/Documents/L3_segmentation/data/dataset_adults'
    dataset = Dataset(path)

    checkpoints_dir = 'checkpoints/'
    name = 'pretrained_19'

    expr_dir = os.path.join(checkpoints_dir, name)

    model = Model(expr_dir, batch_size=64, n_blocks=19, niter=200, niter_decay=200, resume=True)
    training_loader_patches, validation_loader_patches = dataset.get_loaders()
    model.train(training_loader_patches, validation_loader_patches)