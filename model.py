import os
import random
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from torch.utils.tensorboard import SummaryWriter

import network as network
from util import print_log, format_train_log, visualize_training, decode_segmentation


class Model(object):
    def __init__(self, expr_dir, seed=None, batch_size=None,
                 epoch_count=1, niter=150, niter_decay=50, beta1=0.5, lr=0.0002,
                 ngf=64, n_blocks=9, input_nc=1, output_nc=5, use_dropout=True, norm='batch', max_grad_norm=500.,
                 monitor_grad_norm=True, save_epoch_freq=5, print_freq=15, display_epoch_freq=1, testing=False,
                 resume=False):

        self.expr_dir = expr_dir
        self.seed = seed
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

        self.epoch_count = epoch_count
        self.niter = niter
        self.niter_decay = niter_decay
        self.beta1 = beta1
        self.lr = lr
        self.old_lr = self.lr

        self.ngf = ngf
        self.n_blocks = n_blocks
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.use_dropout = use_dropout
        self.norm = norm
        self.max_grad_norm = max_grad_norm

        self.monitor_grad_norm = monitor_grad_norm
        self.save_epoch_freq = save_epoch_freq
        self.print_freq = print_freq
        self.display_epoch_freq = display_epoch_freq
        self.time = datetime.now().strftime("%Y%m%d-%H%M%S")

        # define network we need here
        self.netG = network.define_generator(input_nc=self.input_nc, output_nc=self.output_nc, ngf=self.ngf,
                                             n_blocks=self.n_blocks, use_dropout=self.use_dropout,
                                             device=self.device)

        # define all optimizers here
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=self.lr, betas=(self.beta1, 0.999))

        self.loss = torch.nn.CrossEntropyLoss()

        self.resume = resume

        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        if not os.path.exists(os.path.join(expr_dir, 'TensorBoard')):
            os.makedirs(os.path.join(expr_dir, 'TensorBoard', self.time))

        if not testing:
            num_params = 0
            with open("%s/nets.txt" % self.expr_dir, 'w') as nets_f:
                num_params += network.print_network(self.netG, nets_f)
                nets_f.write('# parameters: %d\n' % num_params)
                nets_f.flush()

        if resume:
            self.load(os.path.join(self.expr_dir, "latest"), True)
            self.netG.to(self.device)

    def train(self, train_dataset, test_set):
        self.batch_size = train_dataset.batch_size
        self.save_options()
        out_f = open(f"{self.expr_dir}/results.txt", 'w')
        use_gpu = torch.cuda.is_available()

        tensorboard_writer = SummaryWriter(os.path.join(self.expr_dir, 'TensorBoard', self.time))

        if self.seed is not None:
            print(f"using random seed: {self.seed}")
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if use_gpu:
                torch.cuda.manual_seed_all(self.seed)

        total_steps = 0
        print_start_time = time.time()

        for epoch in range(self.epoch_count, self.niter + self.niter_decay + 1):
            epoch_start_time = time.time()
            epoch_iter = 0

            for data in train_dataset:
                ct = data['ct'][tio.DATA].to(self.device)
                mask = data['label_map'][tio.DATA].to(self.device)
                ct = ct.transpose_(2, 4)
                mask = torch.squeeze(mask.transpose_(2, 4), 2)
                total_steps += self.batch_size
                epoch_iter += self.batch_size

                if self.monitor_grad_norm:
                    losses, visuals, _ = self.train_instance(ct, mask)
                else:
                    losses, visuals = self.train_instance(ct, mask)

                if total_steps % self.print_freq == 0:
                    t = (time.time() - print_start_time) / self.batch_size
                    print_log(out_f, format_train_log(epoch, epoch_iter, losses, t))
                    tensorboard_writer.add_scalars('Loss', {'train': losses['Training loss']}, total_steps)
                    print_start_time = time.time()

            if epoch % self.display_epoch_freq == 0:
                tensorboard_images = visualize_training(visuals)
                ct = tensorboard_images[0]
                segmentation_mask_decoded = decode_segmentation(tensorboard_images[1])
                fake_segmentation_mask_decoded = decode_segmentation(tensorboard_images[2])

                tensorboard_writer.add_image('CT_train', ct, epoch, epoch_iter / self.batch_size, 'HW')
                tensorboard_writer.add_image('segmentation_mask_train', segmentation_mask_decoded, epoch,
                                             epoch_iter / self.batch_size, 'HWC')
                tensorboard_writer.add_image('fake_segmentation_mask_train', fake_segmentation_mask_decoded, epoch,
                                             epoch_iter / self.batch_size, 'HWC')

            if epoch % self.save_epoch_freq == 0:
                print_log(out_f, 'saving the model at the end of epoch %d, iterations %d' %
                          (epoch, total_steps))
                if not self.resume:
                    self.save('latest')
                else:
                    self.save('latest_resume')

                self.netG.eval()
                total_loss = 0
                with torch.no_grad():
                    for data in test_set:
                        ct = data['ct'][tio.DATA].to(self.device)
                        mask = data['label_map'][tio.DATA].to(self.device)
                        ct = ct.transpose_(2, 4)
                        mask = torch.squeeze(mask.transpose_(2, 4), 2)
                        fake_segmentation = self.netG.forward(ct[0])
                        loss = self.loss(fake_segmentation, mask.to(torch.float32))
                        total_loss += loss
                test_loss = loss.mean()
                print_log(out_f, 'Test loss : %.3f' %
                          test_loss
                          )
                tensorboard_writer.add_scalars('Loss', {'test': test_loss}, total_steps)

                visuals = OrderedDict([('ct', ct.data),
                                       ('segmentation_mask', mask.data),
                                       ('fake_segmentation_mask', fake_segmentation.data)
                                       ])
                tensorboard_images = visualize_training(visuals)
                ct = tensorboard_images[0]
                segmentation_mask_decoded = decode_segmentation(tensorboard_images[1])
                fake_segmentation_mask_decoded = decode_segmentation(tensorboard_images[2])

                tensorboard_writer.add_image('CT_test', ct, epoch, epoch_iter / self.batch_size, 'HW')
                tensorboard_writer.add_image('segmentation_mask_test', segmentation_mask_decoded, epoch,
                                             epoch_iter / self.batch_size, 'HWC')
                tensorboard_writer.add_image('fake_segmentation_mask_test', fake_segmentation_mask_decoded, epoch,
                                             epoch_iter / self.batch_size, 'HWC')

            print_log(out_f, 'End of epoch %d / %d \t Time Taken: %d sec' %
                      (epoch, self.niter + self.niter_decay, time.time() - epoch_start_time))

            if epoch > self.niter:
                self.update_learning_rate()

        out_f.close()
        tensorboard_writer.close()

    def train_instance(self, ct, segmentation):
        fake_segmentation = self.netG.forward(ct[0])
        self.optimizer_G.zero_grad()
        loss = self.loss(fake_segmentation, segmentation.to(torch.float32))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.max_grad_norm)
        self.optimizer_G.step()

        losses = OrderedDict([('Training loss', loss.data.item())])
        visuals = OrderedDict([('ct', ct.data),
                               ('segmentation_mask', segmentation.data),
                               ('fake_segmentation_mask', fake_segmentation.data)
                               ])
        if self.monitor_grad_norm:
            grad_norm = OrderedDict([('grad_norm', grad_norm)])

            return losses, visuals, grad_norm

        return losses, visuals

    def update_learning_rate(self):
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, checkpoint_name):
        checkpoint_path = os.path.join(self.expr_dir, checkpoint_name)
        checkpoint = {
            'netG': self.netG.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path, optimizer=False):
        checkpoint = torch.load(checkpoint_path)
        self.netG.load_state_dict(checkpoint['netG'])
        self.netG.conv_segmentation = nn.Conv2d(self.ngf, self.output_nc, kernel_size=7, padding=3)

        if optimizer:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])

    def eval(self):
        self.netG.eval()

    def save_options(self):
        options_file = open(f"{self.expr_dir}/options.txt", 'wt')
        print_log(options_file, '------------ Options -------------')
        for k, v in sorted(self.__dict__.items()):
            print_log(options_file, '%s: %s' % (str(k), str(v)))
        print_log(options_file, '-------------- End ----------------')

    def test(self, dataset, export_path=None, checkpoint=None, save=False):
        checkpoint = checkpoint or os.path.join(self.expr_dir, "latest")
        self.load(checkpoint)
        self.eval()

        prediction_path = export_path or self.expr_dir
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        start = time.time()
        with torch.no_grad():
            for i, data in enumerate(dataset.loader):
                ct = data['ct'][tio.DATA].to(self.device)
                ct = ct.transpose_(2, 4)
                locations = data[tio.LOCATION]

                fake_segmentation = self.netG.forward(ct)
                fake_segmentation = fake_segmentation.transpose_(2, 4)

                dataset.aggregator.add_batch(fake_segmentation, locations)

                print(f"patch {i + 1}/{len(dataset.loader)}")
            affine = dataset.transform(dataset.subject['ct']).affine
            foreground = dataset.aggregator.get_output_tensor()
            fake_segmentation_mask = foreground.argmax(dim=0, keepdim=True).type(torch.int8)
            prediction = tio.LabelMap(tensor=fake_segmentation_mask, affine=affine)
            print(f"{time.time() - start} sec. for evaluation")
            if save:
                prediction.save(os.path.join(prediction_path, 'fake_segmentation.nii'))
            return prediction
