from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch import nn
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vgg import vgg19_trans
from datasets.crowd_semi import Crowd
from losses.bay_loss_focal import Bay_Loss
from losses.post_gau import Post_Prob
from math import ceil



def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    label = transposed_batch[4]
    return images, points, targets, st_sizes, label


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x, args.info) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate if x == 'train' else default_collate),
                                          batch_size=1,
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=False)
                            for x in ['train', 'val']}
        self.model = vgg19_trans()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.criterion_mse = torch.nn.MSELoss(reduction='sum')
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_all = args.save_all
        self.best_count = 0
        label_count = torch.tensor(
            [0.00016, 0.0048202634789049625, 0.01209819596260786, 0.02164922095835209, 0.03357841819524765,
             0.04810526967048645, 0.06570728123188019, 0.08683456480503082, 0.11207923293113708, 0.1422334909439087,
             0.17838051915168762, 0.22167329490184784, 0.2732916474342346, 0.33556100726127625, 0.41080838441848755,
             0.5030269622802734, 0.6174761652946472, 0.762194037437439, 0.9506691694259644, 1.2056223154067993,
             1.5706151723861694, 2.138580322265625, 3.233219861984253, 7.914860725402832])
        self.label_count = label_count.unsqueeze(1).to(self.device)
        self.mae=[]
        self.mse=[]


    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_epoch(epoch >= args.unlabel_start)
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()


    def train_epoch(self, unlabel):
        epoch_loss = AverageMeter()
        epoch_loss_m = AverageMeter()
        epoch_loss_simi = AverageMeter()
        epoch_loss_semi = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, points, targets, st_sizes, label) in enumerate(self.dataloaders['train']):
            assert inputs.size(0) == 1, 'sorry, the code now only supports one batch'
            if not (unlabel | label[0]):
                continue
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                N = inputs.size(0)
                outputs, mask, feature = self.model(inputs)
                if label[0]:
                    prob_list, gau = self.post_prob(points, st_sizes)
                    gau = gau.flatten().unsqueeze(0).detach() - self.label_count
                    gaum = torch.sum(gau>0, dim=0)
                    gt_bool = gaum.bool()
                    loss = self.criterion(prob_list, targets, outputs)
                    epoch_loss.update(loss.item(), N)
                    loss_m = 0.1 * self.criterion_mse(mask.flatten(), gt_bool.float())

                    epoch_loss_m.update(loss_m.item(), N)
                    loss += loss_m

                else:
                    gau = outputs.flatten().unsqueeze(0).detach() - self.label_count
                    gaum = torch.sum(gau > 0, dim=0)
                    gt_bool = gaum.bool()
                    loss = 0

                fea_sum = torch.sum(feature ** 2, dim=-1) ** 0.5 + 1e-5
                memory_sum = torch.sum(self.model.memory_list ** 2, dim=-1) ** 0.5 + 1e-5
                simi = torch.matmul(feature, self.model.memory_list.T)
                simi = simi / memory_sum / fea_sum.unsqueeze(1)
                loss_simi = torch.sum(torch.mean(simi[~gt_bool] + 1, dim=-1))

                if torch.sum(gt_bool) > 0:
                    gaum = torch.where(gaum > 0, gaum - 1, gaum)[gt_bool]
                    rebalanced = gaum.unsqueeze(1) - torch.arange(len(self.label_count), device=gaum.device).unsqueeze(0)
                    rebalanced = torch.abs(rebalanced)/0.5
                    rebalanced = 2 * torch.exp(-rebalanced) - 1

                    positive = rebalanced>0
                    rebalanced = torch.abs(rebalanced)

                    match_value = simi[gt_bool].gather(1, gaum.unsqueeze(0).long()).T
                    loss_simi += torch.sum(1 - match_value)
                    simi_de = torch.matmul(feature[gt_bool], self.model.memory_list.T.detach())
                    simi_de = simi_de / memory_sum.detach() / fea_sum[gt_bool].unsqueeze(1)

                    soft = torch.exp(simi_de/0.1)
                    soft = soft * rebalanced.detach()
                    soft_loss = torch.sum(soft*positive.float(), dim=-1) / torch.sum(soft, dim=-1)
                    loss_simi += torch.sum(-torch.log(soft_loss+1e-6))

                loss += 0.01 * loss_simi
                if label[0]:
                    epoch_loss_simi.update(0.01 * loss_simi.item(), N)
                else:                 
                    epoch_loss_semi.update(loss.item(), N)
                    loss = loss * 0.05

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count

                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)


        logging.info('Epoch {} Train, Loss: {:.2f}, Loss_m: {:.2f}, Simi: {:.2f}, Semi: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), epoch_loss_m.get_avg(), epoch_loss_simi.get_avg(), epoch_loss_semi.get_avg(),
                             np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            c_size = 2048
            if h >= c_size or w >= c_size:
                h_stride = int(ceil(1.0 * h / c_size))
                w_stride = int(ceil(1.0 * w / c_size))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.model(input)[0]
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_res.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)[0]
                    # save_results(inputs, outputs, self.vis_dir, '{}.jpg'.format(name[0]))
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            if self.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))



