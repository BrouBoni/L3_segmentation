import os

from dataloader import Dataset
from model import Model

if __name__ == '__main__':

    # path = '/home/brouboni/PycharmProjects/L3_segmentation/data/dataset_new/'
    path = '/home/nicoc/Documents/L3_segmentation/data/database_L3_PEPITA'
    dataset = Dataset(path, pretraining=True)

    checkpoints_dir = 'checkpoints/'
    name = 'pretrained_19'

    expr_dir = os.path.join(checkpoints_dir, name)

    model = Model(expr_dir, batch_size=64, n_blocks=19, niter=200, niter_decay=200)
    training_loader_patches, validation_loader_patches = dataset.get_loaders()
    model.train(training_loader_patches, validation_loader_patches)
