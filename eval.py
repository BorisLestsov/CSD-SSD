"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd
from retinanet import build_retinanet

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

from eval_utils import *
from data import detection_collate_eval
from data import coco, voc300, voc512

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--arch',
                    type=str)
parser.add_argument('--trained_model', default='weights/ssd300_COCO_120000.pth',
                    type=str, help='Trained state_dict file path to open')
# parser.add_argument('--trained_model',
#                     default='weights/ssd300_mAP_77.43_v2.pth', type=str,
#                     help='Trained state_dict file path to open')
parser.add_argument('--size', default=300, type=int,
                    help='img size')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=200, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default='/home/soo/data/VOCdevkit/',
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background

    if args.size == 300:
        cfg = voc300
    else:
        cfg = voc512
    if args.arch == "ssd":
        net = build_ssd('test', cfg['min_dim'], cfg['num_classes'], top_k=args.top_k, thresh=args.confidence_threshold)
    elif args.arch == "retinanet":
        net = build_retinanet('test', cfg['min_dim'], cfg['num_classes'], top_k=args.top_k, thresh=args.confidence_threshold)
    else:
        raise Exception("unknown arch")
    checkpoint = torch.load(args.trained_model)
    net.load_state_dict(checkpoint)
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(args.size, dataset_mean),
                           VOCAnnotationTransform(), test=True)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate_eval,
                                  pin_memory=True)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(annopath, imgsetpath, set_type, devkit_path, args.save_folder, net, args.cuda, data_loader,
             BaseTransform(net.size, dataset_mean), args.top_k, args.size,
             thresh=args.confidence_threshold)
