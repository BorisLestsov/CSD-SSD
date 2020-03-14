"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import imagesize
sys.path.append("..")
from utils.augmentations import SSDAugmentation

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


class VOCAnnotationTransform_con(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection_con(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    # image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
    # image_sets = [('2007', 'trainval'), ('2014','ONLY_VOC_IN_COCO')],
    # image_sets = [('2012', 'trainval'), ('2014', 'COCO')],

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform_con(),
                 dataset_name='VOC0712'):
        self.root = root
        self.coco_root = '/home/soo/data/COCO'
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.target_dict = {}
        for (year, name) in image_sets:
            if(name=='trainval'):
                rootpath = osp.join(self.root, 'VOC' + year)
                for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                    self.ids.append((rootpath, line.strip()))
                    img_id = self.ids[-1]
                    if (img_id[0][(len(img_id[0]) - 7):] == 'VOC2007'):
                        target = ET.parse(self._annopath % img_id).getroot()
                        width, height = imagesize.get(self._imgpath % img_id)
                        if self.target_transform is not None:
                            target = self.target_transform(target, width, height)
                    else:
                        target = np.zeros([1, 5])
                    self.target_dict[img_id] = target
            else:
                rootpath = osp.join(self.coco_root)
                for line in open(osp.join(rootpath, name + '.txt')):
                    self.ids.append((rootpath, line.strip()))

                    img_id = self.ids[-1]
                    if (img_id[0][(len(img_id[0]) - 7):] == 'VOC2007'):
                        target = ET.parse(self._annopath % img_id).getroot()
                        width, height = imagesize.get(self._imgpath % img_id)
                        if self.target_transform is not None:
                            target = self.target_transform(target, width, height)
                    else:
                        target = np.zeros([1, 5])
                    self.target_dict[img_id] = target

        weak_list = [
            "ConvertFromInts",
            "ToAbsoluteCoords",
            "RandomMirror",
            "ToPercentCoords",
            "Resize",
            "SubtractMeans",
        ]
        self.weak_transform = SSDAugmentation(self.transform.size, self.transform.mean, weak_list)
        strong_list = [
            "SubtractMeans",
            "PhotometricDistort",
        ]
        self.strong_transform = SSDAugmentation(self.transform.size, (-104, -117, -123), strong_list)

    def __getitem__(self, index):
        # im, gt, h, w = self.pull_item(index)
        im, gt, h, w, semi = self.pull_item(index)

        # return im, gt
        return im, gt, semi

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = self.target_dict[img_id]
        target = np.array(target)
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if (img_id[0][(len(img_id[0]) - 7):] == 'VOC2007'):
            # target = ET.parse(self._annopath % img_id).getroot()
            semi = np.array([1])
            if self.transform is not None:
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                # to rgb
                img = img[:, :, (2, 1, 0)]
                # img = img.transpose(2, 0, 1)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # img = torch.from_numpy(img).permute(2, 0, 1)
            img = np.concatenate([img[None, ...], img[None, ...]], axis=0)
            img = torch.from_numpy(img).permute(0, 3, 1, 2)

        elif (img_id[0][(len(img_id[0]) - 7):] == 'VOC2012'):
            semi = np.array([0])
            if self.weak_transform is not None:
                img_w, boxes, labels = self.weak_transform(img, target[:, :4], target[:, 4])
                # to rgb
                img_w = img_w[:, :, (2, 1, 0)]
                # img = img.transpose(2, 0, 1)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            if self.strong_transform is not None:
                img_s, boxes, labels = self.strong_transform(img_w.copy(), target[:, :4], target[:, 4])
                # to rgb
                img_s = img_s[:, :, (2, 1, 0)]
                # img = img.transpose(2, 0, 1)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            img = np.concatenate([img_w[None, ...], img_s[None, ...]], axis=0)
            img = torch.from_numpy(img).permute(0, 3, 1, 2)
        else:
            raise Exception("unknown dataset?")
            #semi = np.array([0])

        return img, target, height, width, semi

        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
