import torch
import numpy as np
from functools import partial
from timm.data.loader import _worker_init
from timm.data.distributed_sampler import OrderedDistributedSampler

try:
    from datasets.transforms import *
except:
    from transforms import *
from .mp_liver_dataset import MultiPhaseLiverDataset


class MultiPhaseLiverDataset2d(MultiPhaseLiverDataset):
    def __init__(self, args, is_training=True,num_of_copies =4):
        super().__init__(args, is_training)
        self.num_of_copies = 4

    def __getitem__(self, index):
        args = self.args
        if self.is_training:
            type_choice = np.random.randint(self.lab_numpy.max() + 1)
            possble_loc = np.where(self.lab_numpy == type_choice)[0]
            index = possble_loc[np.random.permutation(possble_loc.shape[0])[0]]
        image = self.load_mp_images(self.img_list[index])
        if self.is_training:
            image = self.transforms(image, args.train_transform_list)
        else:
            image = self.test_time_transforms(image, args.val_transform_list,num_of_copies=self.num_of_copies)
        image = image.copy()
        if self.is_training:
            i = np.random.randint(3,9)
        else:
            i = 6
        image = image[:,i:i+3,:,:]
        label = int(self.lab_list[index])
        # if label in [1,3,6]:
        #     label =1
        # else:
        #     label = 0
        return (image, label)

    def load_mp_images(self, mp_img_list):
        mp_image = []
        for img in mp_img_list:
            image = load_nii_file(img)
            mp_image.append(image[None, ...])
        mp_image = np.concatenate(mp_image, axis=0)
        return mp_image

    def test_time_transforms(self, mp_image, transform_list, num_of_copies):
        args = self.args
        if num_of_copies == 1:
            return self.transforms(mp_image, transform_list)
        elif num_of_copies > 1:
            aug_image = np.zeros(
                [num_of_copies,mp_image.shape[0], args.crop_size[0], args.crop_size[1], args.crop_size[2]],
                dtype=mp_image.dtype)
            for i in range(num_of_copies):
                if i == 0:
                    aug_image[i,:,:,:,:]= self.transforms(mp_image, transform_list)
                else:
                    tmp_list = ['resize3D']
                    if np.random.uniform(0,1) > 0.5:
                        tmp_list.append('z_flip')
                    if np.random.uniform(0, 1) > 0.5:
                        tmp_list.append('x_flip')
                    if np.random.uniform(0, 1) > 0.5:
                        tmp_list.append('y_flip')
                    # if np.random.uniform(0, 1) > 0.5:
                    #     tmp_list.append('rotation')
                    # if np.random.uniform(0, 1) > 0.5:
                    #     tmp_list.append('random_crop')
                    if 'random_crop' not in tmp_list:
                        tmp_list.append('center_crop')
                    aug_image[i, :, :, :, :] = self.transforms(mp_image, tmp_list)
        aug_image = aug_image.reshape(-1,aug_image.shape[2],aug_image.shape[3],aug_image.shape[4]).astype(np.float32)
        return aug_image



    def transforms(self, mp_image, transform_list):
        args = self.args
        if 'resize3D' in transform_list:
            mp_image = resize3D(mp_image, args.img_size)
        if 'resize3D_mod' in transform_list:
            mp_image = resize3D_mod(mp_image, args.img_size)
        if 'center_crop' in transform_list:
            if self.is_training:
                mp_image = center_crop(mp_image, args.img_size)
            else:
                mp_image = center_crop(mp_image, args.crop_size)
        if 'random_crop' in transform_list:
            mp_image = random_crop(mp_image, args.crop_size)
        if 'z_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='z', p=args.flip_prob)
        if 'x_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='x', p=args.flip_prob)
        if 'y_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='y', p=args.flip_prob)
        if 'rotation' in transform_list:
            mp_image = rotate(mp_image, args.angle)
        return mp_image

    def __len__(self):
        return len(self.img_list)


def create_loader(
        dataset=None,
        batch_size=1,
        is_training=False,
        num_aug_repeats=0,
        num_workers=1,
        distributed=False,
        collate_fn=None,
        pin_memory=False,
        persistent_workers=True,
        worker_seeding='all',
):
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    return loader


if __name__ == "__main__":
    import yaml
    import parser
    import argparse
    from tqdm import tqdm

    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        '--data_dir', default='data/classification_dataset/images/', type=str)
    parser.add_argument(
        '--train_anno_file', default='data/classification_dataset/labels/train_fold1.txt', type=str)
    parser.add_argument(
        '--val_anno_file', default='data/classification_dataset/labels/val_fold1.txt', type=str)
    parser.add_argument('--train_transform_list', default=['random_crop',
                                                           'z_flip',
                                                           'x_flip',
                                                           'y_flip',
                                                           'rotation', ],
                        nargs='+', type=str)
    parser.add_argument('--val_transform_list',
                        default=['center_crop'], nargs='+', type=str)
    parser.add_argument('--img_size', default=(16, 128, 128),
                        type=int, nargs='+', help='input image size.')
    parser.add_argument('--crop_size', default=(14, 112, 112),
                        type=int, nargs='+', help='cropped image size.')
    parser.add_argument('--flip_prob', default=0.5, type=float,
                        help='Random flip prob (default: 0.5)')
    parser.add_argument('--angle', default=45, type=int)


    def _parse_args():
        # Do we have a config file to parse?
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)
        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text


    args, args_text = _parse_args()
    args_text = yaml.load(args_text, Loader=yaml.FullLoader)
    args_text['img_size'] = 'xxx'
    print(args_text)

    args.distributed = False
    args.batch_size = 100

    dataset = MultiPhaseLiverDataset(args, is_training=True)
    data_loader = create_loader(dataset, batch_size=3, is_training=True)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=3)
    for images, labels in data_loader:
        print(images.shape)
        print(labels)

    # val_dataset = MultiPhaseLiverDataset(args, is_training=False)
    # val_data_loader = create_loader(val_dataset, batch_size=10, is_training=False)
    # for images, labels in val_data_loader:
    #     print(images.shape)
    #     print(labels)
