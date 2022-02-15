import numpy as np
from pathlib import Path
from PIL import Image
import datasets.transforms as T
import torch.utils.data


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, images, classes, class_anno, box_anno, split_info, split, transforms=None):
        self.img_dir = img_folder
        self.classes = self.get_cub_classes(classes)
        self.image_ids = self.get_image_list(split_info, split)
        self.image_names = self.get_image_names(images)
        self.image_labels = self.get_image_labels(class_anno)
        self.boxes = self.get_boxes(box_anno)
        self._transforms = transforms

    def get_cub_classes(self, class_file):
        class_list = [None, ]
        with open(class_file, 'r') as fp:
            lines = fp.readlines()
            list(map(lambda x: class_list.append(x.strip('\n').split(' ')[-1]), lines))
        return class_list

    def get_image_list(self, split_info, split):
        image_list = []
        # Using convention in the split file of CUB
        if split == 'train':
            split = '1'
        else:
            split = '0'
        with open(split_info, 'r') as fp:
            lines = fp.readlines()
            list(map(lambda x: image_list.append(int(x.strip('\n').split(' ')[0])) if x.strip('\n')[-1] == split else 1,
                     lines))
        return image_list

    def get_image_names(self, images):
        image_names = []
        with open(images, 'r') as fp:
            lines = fp.readlines()
            list(map(lambda x: image_names.append(lines[x - 1].strip('\n').split(' ')[-1]), self.image_ids))
        return image_names

    def get_image_labels(self, class_anno):
        image_labels = []
        with open(class_anno, 'r') as fp:
            lines = fp.readlines()
            list(map(lambda x: image_labels.append(int(lines[x - 1].strip('\n').split(' ')[-1])), self.image_ids))
        return image_labels

    def get_boxes(self, box_anno):
        boxes = dict()
        with open(box_anno, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                data = line.strip('\n').split(' ')
                boxes[int(data[0])] = list(map(float, data[1:]))
        return boxes

    def __getitem__(self, idx):
        img = Image.open(f'{self.img_dir}/{self.image_names[idx]}').convert('RGB')
        target_label = self.image_labels[idx]
        target = dict()
        target['labels'] = torch.Tensor(np.asarray([target_label]).astype(np.int64)).long()
        target['img_id'] = torch.Tensor(np.asarray([self.image_ids[idx]]).astype(np.int64)).long()
        target['boxes'] = torch.Tensor(np.asarray([self.boxes[self.image_ids[idx]]]))
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # img = np.asarray(img)
        return img, target

    def __len__(self):
        return len(self.image_ids)


def make_cub_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided CUB path {root} does not exist'
    class_list = f'{args.coco_path}/classes.txt'
    class_anno = f'{args.coco_path}/image_class_labels.txt'
    box_anno = f'{args.coco_path}/bounding_boxes.txt'
    img_folder = f'{args.coco_path}/images'
    split_info = f'{args.coco_path}/train_test_split.txt'
    images = f'{args.coco_path}/images.txt'
    dataset = CUBDataset(img_folder, images=images, classes=class_list, class_anno=class_anno, box_anno=box_anno,
                         split_info=split_info, split=image_set, transforms=make_cub_transforms(image_set))
    return dataset