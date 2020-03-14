from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
# from ssd_consistency import build_ssd_con
from csd import build_ssd_con
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import math
import copy

from eval_utils import test_net

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--arch',
                    type=str)
parser.add_argument('--use_focal',
                    type=str2bool)
parser.add_argument('--dataset', default='VOC300', choices=['VOC300', 'VOC512'],
                    type=str, help='VOC300 or VOC512')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,  # None  'weights/ssd300_COCO_80000.pth'
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--size', default=30000, type=int,
                    help='img size')
parser.add_argument('--log_period', default=20, type=int,
                    help='log pediod')
parser.add_argument('--eval_period', default=5000, type=int,
                    help='eval period')
parser.add_argument('--checkpoint_period', default=5000, type=int,
                    help='checkpoint period')
parser.add_argument('--save_folder_eval', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--top_k', default=200, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--need_vis', default=False, type=str2bool,
                    help='vis')
args = parser.parse_args()

# EVAL UTILS
annopath = os.path.join(args.dataset_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.dataset_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.dataset_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.dataset_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


viz = None
def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC300':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc300
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'VOC512':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc512
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom()
    epoch_size = len(dataset) // args.batch_size

    finish_flag = True

    while(finish_flag):
        ssd_net = build_ssd_con('train', cfg['min_dim'], cfg['num_classes'], top_k=args.top_k, thresh=args.confidence_threshold)
        net = ssd_net

        if args.cuda:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True

        if args.resume:
            print('Resuming training, loading {}...'.format(args.resume))
            ssd_net.load_weights(args.resume)
        else:
            vgg_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network...')
            ssd_net.vgg.load_state_dict(vgg_weights)

        if args.cuda:
            net = net.cuda()

        if not args.resume:
            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                 False, args.cuda)
        conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()


        net.train()
        # loss counters
        loc_loss = 0
        conf_loss = 0
        epoch = 0
        supervised_flag = 1
        print('Loading the dataset...')

        step_index = 0


        if args.visdom:
            vis_title = 'SSD.PyTorch on ' + dataset.name
            vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
            iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
            epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)


        total_un_iter_num = 0


        supervised_batch =  args.batch_size
        #unsupervised_batch = args.batch_size - supervised_batch
        #data_shuffle = 0


        val_dataset = VOCDetection(args.dataset_root, [('2007', "test")],
                                BaseTransform(cfg["min_dim"], dataset_mean),
                                VOCAnnotationTransform(), test=True)
        val_data_loader = data.DataLoader(val_dataset, args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False, collate_fn=detection_collate_eval,
                                    pin_memory=True)
        if(args.start_iter==0):
            supervised_dataset = VOCDetection_con_init(root=args.dataset_root,
                                                     transform=SSDAugmentation(cfg['min_dim'],
                                                                                  MEANS))
        else:
            supervised_flag = 0
            supervised_dataset = VOCDetection_con(root=args.dataset_root,
                                                     transform=SSDAugmentation(cfg['min_dim'],
                                                                                  MEANS))#,shuffle_flag=data_shuffle)
            #data_shuffle = 1

        supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch,
                                                   num_workers=args.num_workers,
                                                   shuffle=True, collate_fn=detection_collate,
                                                   pin_memory=True, drop_last=True)


        batch_iterator = iter(supervised_data_loader)

        mean_aps = []
        for iteration in range(args.start_iter, cfg['max_iter']):
            if (iteration % args.eval_period == 0) and iteration >= 2000:
                print("EVAL")
                net.eval()
                net.module.phase = "test"
                mean_ap = test_net(annopath, imgsetpath, set_type, devkit_path, args.save_folder_eval, net, args.cuda, val_data_loader,
                                BaseTransform(cfg["min_dim"], dataset_mean), args.top_k, cfg["min_dim"],
                                thresh=args.confidence_threshold)
                mean_aps.append(mean_ap)
                net.train()
                net.module.phase = "train"
                if len(mean_aps) >= 4:
                    print("check ap")
                    if mean_ap <= mean_aps[-2] and mean_ap <= mean_aps[-3] and mean_ap <= mean_aps[-4]:
                        print("dropping LR: ", mean_aps[-2], mean_aps[-3], mean_aps[-4])
                        mean_aps = []
                        step_index += 1
                        if step_index > 3:
                            print("FIN")
                            finish_flag = False
                            break
                        adjust_learning_rate(optimizer, args.gamma, step_index)

            if args.visdom and iteration != 0 and (iteration % args.epoch_size == 0):
                # update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                #                 'append', args.epoch_size)
                # reset epoch loss counters
                loc_loss = 0
                conf_loss = 0
                epoch += 1

            # if iteration in cfg['lr_steps']:
            #     step_index += 1
            #     adjust_learning_rate(optimizer, args.gamma, step_index)

            try:
                images, targets, semis = next(batch_iterator)
            except StopIteration:
                supervised_flag = 0
                supervised_dataset = VOCDetection_con(root=args.dataset_root,
                                                         transform=SSDAugmentation(cfg['min_dim'],
                                                                                      MEANS))#, shuffle_flag=data_shuffle)
                supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch,
                                                           num_workers=args.num_workers,
                                                           shuffle=True, collate_fn=detection_collate,
                                                           pin_memory=True, drop_last=True)
                batch_iterator = iter(supervised_data_loader)
                images, targets, semis = next(batch_iterator)


            # TRAINING
            if not supervised_flag:
                sup_image_binary_index = []
                unsup_weak_image_binary_index = []
                unsup_strong_image_binary_index = []
                all_images = []
                targets_dbg = []
                for el_idx, is_sup in enumerate(semis):
                    if(int(is_sup)==0):
                        # unsup
                        all_images.append(images[el_idx, 0, ...].unsqueeze(0))
                        all_images.append(images[el_idx, 1, ...].unsqueeze(0))
                        unsup_weak_image_binary_index += [1, 0]
                        unsup_strong_image_binary_index += [0, 1]
                        sup_image_binary_index += [0, 0]
                        targets_dbg.append(targets[el_idx])
                        targets_dbg.append(targets[el_idx])
                    else:
                        # sup
                        all_images.append(images[el_idx, 0, ...].unsqueeze(0))
                        all_images.append(images[el_idx, 0, ...].unsqueeze(0))
                        unsup_weak_image_binary_index += [0, 0]
                        unsup_strong_image_binary_index += [0, 0]
                        sup_image_binary_index += [1, 0]
                        targets_dbg.append(targets[el_idx])
                        targets_dbg.append(targets[el_idx])

                targets = [tgt for tgt_i, tgt in enumerate(targets) if semis[tgt_i] == 1.]

                sup_image_binary_index = np.array(sup_image_binary_index)
                unsup_weak_image_binary_index = np.array(unsup_weak_image_binary_index)
                unsup_strong_image_binary_index = np.array(unsup_strong_image_binary_index)

                images = torch.cat(all_images, dim=0)

                # print([1 if sss == 1 else 0 for sss in semis])
                # print(sup_image_binary_index)
                # print(unsup_weak_image_binary_index)
                # print(unsup_strong_image_binary_index)
                # print(images.shape)

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()

            out, conf, conf_flip, loc, loc_flip = net(images)
            loc_data, conf_data, priors = out


            # PS
            if not supervised_flag:
                loss_l_ps = Variable(torch.cuda.FloatTensor([0]))
                loss_c_ps = Variable(torch.cuda.FloatTensor([0]))
                train_pseudo = False

                net.eval()
                net.module.phase = "ps"
                with torch.no_grad():
                    detections_batch = net(images.cuda()).data
                net.train()
                net.module.phase = "train"

                ps_thresh = 0.5
                all_boxes = [[] for _ in range(images.shape[0])]
                h, w = images.shape[2], images.shape[3]
                all_cls_dets = []
                for im_i in range(images.shape[0]):
                    detections = detections_batch[im_i].unsqueeze(0)
                    for j in range(1, detections.size(1)):
                        dets = detections[0, j, :]
                        mask = dets[:, 0].gt(ps_thresh).expand(5, dets.size(0)).t()
                        dets = torch.masked_select(dets, mask).view(-1, 5)
                        if dets.dim() == 0:
                            cls_dets = np.zeros(0, 5)
                        else:
                            boxes = dets[:, 1:]
                            scores = dets[:, 0].cpu().numpy()
                            cls_dets = np.hstack((boxes.cpu().numpy(),
                                                np.full((boxes.shape[0], 1), j-1))).astype(np.float32,
                                                                                copy=False)
                        all_boxes[im_i].append(cls_dets)
                all_boxes = [np.vstack(boxes) for boxes in all_boxes]
                targets_pred = [torch.from_numpy(boxes) for boxes in all_boxes]

                targets_weak_ind    = unsup_weak_image_binary_index
                targets_nz_ind = np.array([1 if targets_pred[img_ind].shape[0] !=0 else 0 for img_ind in range(images.shape[0])])

                targets_weak_nz_ind = torch.from_numpy(targets_weak_ind & targets_nz_ind)
                train_pseudo = targets_weak_nz_ind.bool().any()
                if train_pseudo:
                    targets_weak = [torch.from_numpy(np.array(targets_pred[img_ind])) for img_ind in range(images.shape[0]) if targets_weak_nz_ind[img_ind] == 1]
                    targets_ps = [Variable(ann.detach().cuda(), volatile=True) for ann in targets_weak]

                    strong_index = np.where(targets_weak_nz_ind == 1)[0] + 1
                    output_strong = (
                        loc_data[strong_index, ...],
                        conf_data[strong_index, ...],
                        priors
                    )
                    loss_l_ps, loss_c_ps = criterion(output_strong, targets_ps)


                if args.need_vis:
                    # sup
                    img_set = images
                    for dbg_i in range(img_set.shape[0]):
                        dbg_img = img_set[dbg_i].cpu().numpy().transpose(1,2,0)
                        dbg_img[:,:,0] += 104
                        dbg_img[:,:,1] += 113
                        dbg_img[:,:,2] += 123
                        dbg_img = np.clip(dbg_img, 0, 255)
                        dbg_img = dbg_img.astype(np.uint8).copy()

                        if sup_image_binary_index[dbg_i]:
                            dbg_tgt = targets_dbg[dbg_i].cpu().numpy().copy()
                            dbg_tgt[:, 0] *= dbg_img.shape[1]
                            dbg_tgt[:, 2] *= dbg_img.shape[0]
                            dbg_tgt[:, 1] *= dbg_img.shape[1]
                            dbg_tgt[:, 3] *= dbg_img.shape[0]
                            dbg_tgt = dbg_tgt.astype(np.int32)
                            for pt in dbg_tgt:
                                cv2.rectangle(dbg_img, (pt[0], pt[1]), (pt[2], pt[3]), (255,0,0), 2)

                        if unsup_weak_image_binary_index[dbg_i] or unsup_strong_image_binary_index[dbg_i]:
                            dbg_pred = targets_pred[dbg_i].cpu().numpy().copy()
                            if dbg_pred.size != 0:
                                dbg_pred[:, 0] *= dbg_img.shape[1]
                                dbg_pred[:, 2] *= dbg_img.shape[0]
                                dbg_pred[:, 1] *= dbg_img.shape[1]
                                dbg_pred[:, 3] *= dbg_img.shape[0]
                                for pt in dbg_pred:
                                    cv2.rectangle(dbg_img, (pt[0], pt[1]), (pt[2], pt[3]), (0,0,255), 2)

                        print("saving...")
                        cv2.imwrite("dbg/out_{}_{}.png".format(iteration, dbg_i), cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB))


            sup_image_index = np.where(sup_image_binary_index == 1)[0]
            unsup_image_index = np.where(sup_image_binary_index == 0)[0]


            if (len(sup_image_index) != 0):
                loc_data = loc_data[sup_image_index,:,:]
                conf_data = conf_data[sup_image_index,:,:]
                output = (
                    loc_data,
                    conf_data,
                    priors
                )

            # backprop
            # loss = Variable(torch.cuda.FloatTensor([0]))
            loss_l = Variable(torch.cuda.FloatTensor([0]))
            loss_c = Variable(torch.cuda.FloatTensor([0]))



            if(len(sup_image_index)!=0):
                try:
                    loss_l, loss_c = criterion(output, targets)
                except:
                    break
                    print('--------------')


            sampling = True
            if(sampling is True):
                conf_class = conf[:,:,1:].clone()
                background_score = conf[:, :, 0].clone()
                each_val, each_index = torch.max(conf_class, dim=2)
                mask_val = each_val > background_score
                mask_val = mask_val.data

                mask_conf_index = mask_val.unsqueeze(2).expand_as(conf)
                mask_loc_index = mask_val.unsqueeze(2).expand_as(loc)

                conf_mask_sample = conf.clone()
                loc_mask_sample = loc.clone()
                conf_sampled = conf_mask_sample[mask_conf_index].view(-1, 21)
                loc_sampled = loc_mask_sample[mask_loc_index].view(-1, 4)

                conf_mask_sample_flip = conf_flip.clone()
                loc_mask_sample_flip = loc_flip.clone()
                conf_sampled_flip = conf_mask_sample_flip[mask_conf_index].view(-1, 21)
                loc_sampled_flip = loc_mask_sample_flip[mask_loc_index].view(-1, 4)

            if(mask_val.sum()>0):
                ## JSD !!!!!1
                conf_sampled_flip = conf_sampled_flip + 1e-7
                conf_sampled = conf_sampled + 1e-7
                consistency_conf_loss_a = conf_consistency_criterion(conf_sampled.log(), conf_sampled_flip.detach()).sum(-1).mean()
                consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(), conf_sampled.detach()).sum(-1).mean()
                consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b

                ## LOC LOSS
                consistency_loc_loss_x = torch.mean(torch.pow(loc_sampled[:, 0] + loc_sampled_flip[:, 0], exponent=2))
                consistency_loc_loss_y = torch.mean(torch.pow(loc_sampled[:, 1] - loc_sampled_flip[:, 1], exponent=2))
                consistency_loc_loss_w = torch.mean(torch.pow(loc_sampled[:, 2] - loc_sampled_flip[:, 2], exponent=2))
                consistency_loc_loss_h = torch.mean(torch.pow(loc_sampled[:, 3] - loc_sampled_flip[:, 3], exponent=2))

                consistency_loc_loss = torch.div(
                    consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
                    4)

            else:
                consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
                consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))

            consistency_loss = torch.div(consistency_conf_loss,2) + consistency_loc_loss

            ramp_weight = rampweight(iteration)
            consistency_loss = torch.mul(consistency_loss, ramp_weight)


            if(supervised_flag ==1):
                loss = loss_l + loss_c + consistency_loss
            else:
                if(len(sup_image_index)==0):
                    loss = consistency_loss
                else:
                    loss = loss_l + loss_c + consistency_loss
                # PSEUDO LABEL
                if train_pseudo:
                    loss += loss_l_ps + loss_c_ps


            if(loss.data>0):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t1 = time.time()
            if(len(sup_image_index)==0):
                loss_l.data = Variable(torch.cuda.FloatTensor([0]))
                loss_c.data = Variable(torch.cuda.FloatTensor([0]))
            else:
                loc_loss += loss_l.data  # [0]
                conf_loss += loss_c.data  # [0]


            if iteration % args.log_period == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f || consistency_loss : %.4f ||' % (loss.data, consistency_loss.data), end=' ')
                print('loss: %.4f , loss_c: %.4f , loss_l: %.4f , loss_con: %.4f, loss_c_ps: %.4f, loss_l_ps: %.4f, lr : %.4f, super_len : %d\n' % (loss.data, loss_c.data, loss_l.data, consistency_loss.data, loss_c_ps, loss_l_ps, float(optimizer.param_groups[0]['lr']),len(sup_image_index)))


            if(float(loss)>100):
                print("WTF BIG LOSS", float(loss))
                break

            if args.visdom and iteration != 0:
                update_vis_plot(iteration, loss_l.data, loss_c.data,
                                iter_plot, epoch_plot, 'append')

            if iteration != 0 and (iteration+1) % args.checkpoint_period == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                           repr(iteration+1) + '.pth')
        # torch.save(ssd_net.state_dict(), args.save_folder + '' + args.dataset + '.pth')
        print('-------------------------------\n')
        print(loss.data)
        print('-------------------------------')

        if((iteration +1) ==cfg['max_iter']):
            finish_flag = False


def rampweight(iteration):
    ramp_up_end = 32000
    ramp_down_start = 100000

    if(iteration<ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end),2))
    elif(iteration>ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (120000 - iteration) / 20000),2)) 
    else:
        ramp_weight = 1 


    if(iteration==0):
        ramp_weight = 0

    return ramp_weight




def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        if not window2 is None:
            viz.line(
                X=torch.zeros((1, 3)).cpu(),
                Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
                win=window2,
                update=True
            )


if __name__ == '__main__':
    train()
