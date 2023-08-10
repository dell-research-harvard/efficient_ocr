import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
import os
import json

from .samplers import *
from .transforms import *
from datetime import datetime
from tqdm import tqdm
import random


def diff_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


class CustomSubset(Dataset):


    def __init__(self, dataset, indices):
        self.super_dataset = dataset
        self.indices = indices
        self.class_to_idx = dataset.class_to_idx
        self.targets_dict = {idx: target for idx, target in tqdm(enumerate(dataset.targets))}
        self.path_dict = {idx: path for idx, path in tqdm(enumerate(dataset.data))}


    def __getitem__(self, idx):
        image, target = self.super_dataset[self.indices[idx]]
        return (image, target)


    def __len__(self):
        return len(self.indices)


    @property
    def targets(self):
        return [self.targets_dict[idx] for idx in self.indices]
    

    @property
    def paths(self):
        return [self.path_dict[idx] for idx in self.indices]
    

    @property
    def data(self):
        return [self.super_dataset.data[idx] for idx in self.indices]
    

    @property
    def subsetted_targets_dict(self):
        """Only return target dict for subsetted self.indices. Use the targets property instead"""
        new_targets_dict = {new_idx: self.targets_dict[idx] for new_idx, idx in enumerate(self.indices)}
        return new_targets_dict

        
    @property
    def subsetted_path_dict(self):
        """Only return path dict for subsetted self.indices"""
        new_path_dict = {new_idx: self.path_dict[idx] for new_idx, idx in enumerate(self.indices)}
        return new_path_dict

  
class FontImageFolder(ImageFolder):
    """
    Expand factor is a beta feature that expands the dataset by a factor of expand_factor. Use at your own risk. 
    """


    def __init__(
        self, root, render_transform=None, paired_transform=None, patch_resize=False,
        loader=default_loader, is_valid_file=None, expand_factor=1 
    ):
        super(FontImageFolder, self).__init__(
            root, loader, IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file=is_valid_file
        )
        self.data = self.samples
        self.render_transform = render_transform
        self.paired_transform = paired_transform
        self.patch_resize = patch_resize
        self.expand_factor = expand_factor
        ###expand the self.data list by the expand factor
        self.data = self.data * self.expand_factor
        ##expand the targets list by the expand factor
        self.targets = self.targets * self.expand_factor
        ###modufy class to idx
        self.class_to_idx = {k: v for k, v in self.class_to_idx.items() for _ in range(self.expand_factor)}


    def __getitem__(self, index):
        path, target = self.data[index] # Wrap around for indexing if expand_factor > 1

        sample = self.loader(path)

        if os.path.basename(path).startswith("PAIRED") and not ("sn-" in os.path.basename(path)) and (
            self.paired_transform is not None or self.render_transform is not None
        ):
            sample = self.paired_transform(sample)
        else:
            try:
                sample = self.render_transform(sample)
            except RuntimeError:
                sample = self.paired_transform(sample)

        return sample, target


    def __len__(self):
        return len(self.data)


def create_dataset(
        root_dir, 
        train_ann_path,
        val_ann_path,
        test_ann_path, 
        batch_size,
        train_mode="char",
        hardmined_txt=None, 
        m=4,
        finetune=False,
        pretrain=False,
        high_blur=False,
        latin_suggested_augs=False,
        char_trans_version=4,
        diff_sizes=False,
        imsize=224,
        num_passes=1,
        no_aug=False,
        k=8,
        aug_paired=False,
        expansion_factor=1,
        tvt_split=[0.7, 0.15, 0.15],
    ):

    if finetune and pretrain:
        raise NotImplementedError
    if finetune:
        print("Finetuning mode!")
    if pretrain:
        print("Pretraining model!")

    start_time = datetime.now()

    if train_mode == "char":

        dataset = FontImageFolder(
            root_dir, 
            render_transform= create_paired_transform_char(size=imsize) if no_aug else \
                create_render_transform_char(char_trans_version, latin_suggested_augs, size=imsize), 
            paired_transform=create_paired_transform_char(size=imsize),
            patch_resize=diff_sizes,
        expand_factor=expansion_factor)

    else:

        dataset = FontImageFolder(
            root_dir, 
            render_transform= create_paired_transform(size=imsize) if no_aug else \
                create_render_transform(high_blur, size=imsize), 
            paired_transform=create_paired_transform(size=imsize) if not aug_paired else create_render_transform(high_blur, size=imsize) ,
            patch_resize=diff_sizes,
        expand_factor=expansion_factor)

    end_time = datetime.now()
    print(f"Dataset creation time: {end_time - start_time}")

    with open(train_ann_path) as f: 
        train_ann = json.load(f)
        train_stems = [os.path.splitext(x['file_name'])[0].split("/")[-1] for x in train_ann['images']]
    train_stems = set(train_stems)
    with open(val_ann_path) as f: 
        val_ann = json.load(f)
        val_stems = [os.path.splitext(x['file_name'])[0].split("/")[-1] for x in val_ann['images']]
    with open(test_ann_path) as f: 
        test_ann = json.load(f)
        test_stems = [os.path.splitext(x['file_name'])[0].split("/")[-1] for x in test_ann['images']]

    ###Make stems into sets for faster lookup
    val_stems = set(val_stems)
    test_stems = set(test_stems)

    print("Start indexing...")
    paired_val_idx = [idx for idx, (p, t) in (enumerate(dataset.data)) if \
        os.path.basename(p).split(".")[0] in (val_stems)]
    paired_test_idx =[idx for idx, (p, t) in (enumerate(dataset.data)) if \
        os.path.basename(p).split(".")[0] in (test_stems)]
    paired_train_idx = [idx for idx, (p, t) in (enumerate(dataset.data)) if \
        os.path.basename(p).split(".")[0] in (train_stems)]
    render_idx = [idx for idx, (p, t) in enumerate(dataset.data) if \
        not os.path.basename(p).startswith("PAIRED")]
    print("End indexing...")
    
    if train_mode == "char":
        other_idx = [idx for idx, (p, t) in enumerate(dataset.data) if \
            not idx in paired_train_idx + paired_val_idx + paired_test_idx + render_idx]
        print("Total other idx: ", len(other_idx))
    
    print(f"Train len: {len(paired_train_idx)}\nVal len: {len(paired_val_idx)}\nTest len: {len(paired_test_idx)}")
    assert len(set(paired_train_idx).intersection(set(paired_val_idx))) == 0
    assert len(set(paired_val_idx).intersection(set(paired_test_idx))) == 0
    assert len(set(paired_test_idx).intersection(set(paired_train_idx))) == 0
    
    """
    if len(other_idx) != 0:
        other_len = len(other_idx)
        other_train_end_idx = int(other_len * tvt_split[0])
        other_val_end_idx = int(other_len * (tvt_split[0]+tvt_split[1]))
        random.seed(99)
        other_idx = random.sample(other_idx, other_len)
        other_train_idx = other_idx[:other_train_end_idx]
        other_val_idx = other_idx[other_train_end_idx:]
        other_test_idx = other_idx[other_val_end_idx:]
        paired_train_idx += other_train_idx
        paired_val_idx += other_val_idx
        paired_test_idx += other_test_idx
    """
    
    print("Total render idx: ", len(render_idx))

    train_stems = list(train_stems)

    ##Revert to lists
    val_stems = list(val_stems)
    test_stems = list(test_stems)

    print("Time to create indices: ", datetime.now() - end_time)
    print(f"Train len: {len(paired_train_idx)}")
    print(f"Tender len: {len(render_idx)}")
    
    if finetune:
        idx_train = sorted(paired_train_idx)
    elif pretrain:
        idx_train = sorted(render_idx)
    else:
        idx_train = sorted(render_idx + paired_train_idx)
        ##Dedup
        idx_train = list(set(idx_train))

    train_dataset = CustomSubset(dataset, idx_train)

    idx_val = sorted(list(set(paired_val_idx)))
    idx_test = sorted(list(set(paired_test_idx))) 
    val_dataset = CustomSubset(dataset, idx_val)
    test_dataset = CustomSubset(dataset, idx_test)

    #####Flag - Add some checks here in later version to test for overlaps.

    print(f"Len train dataset: {len(train_dataset)}")
    print("Time to create subsets: ", datetime.now() - start_time)

    if hardmined_txt is None:
        train_sampler = NoReplacementMPerClassSampler(
            train_dataset, m=m, batch_size=batch_size, num_passes=num_passes
        )
    else:
        print("Using hard negatives!")

        with open(hardmined_txt) as f:
            hard_negatives = f.read().split("\n")
            print(f"Len hard negatives: {len(hard_negatives)}")
            if train_mode == "char":
                train_sampler = HardNegativeClassSamplerChar(train_dataset, 
                    train_dataset.class_to_idx, hard_negatives, m=m, batch_size=batch_size, 
                    num_passes=num_passes
                )
            else:
                train_sampler = AllHNSamplerSplitBatchesPairRender(train_dataset, 
                    train_dataset.class_to_idx, hns_set_size=k, m=m, batch_size=batch_size, 
                    num_passes=num_passes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, drop_last=True, 
        sampler=train_sampler, collate_fn=diff_size_collate if diff_sizes else None)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=False,
        collate_fn=diff_size_collate if diff_sizes else None)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=False,
        collate_fn=diff_size_collate if diff_sizes else None)

    print("Time to create dataloaders: ", datetime.now() - start_time)

    print("Train dataset size: ", len(train_dataset))

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, train_loader.sampler.nbatches  if (hardmined_txt!=None and train_mode=="word") else None


def create_paired_dataset(root_dir,train_mode="char", imsize=224):

    if train_mode == "char":
        paired_transform = render_transform = create_paired_transform_char(imsize)
    else:
        paired_transform = render_transform = create_paired_transform(imsize)
    dataset = FontImageFolder(root_dir, render_transform=render_transform, paired_transform=paired_transform)
    idx_paired = [idx for idx, (p, t) in enumerate(dataset.data) if os.path.basename(p).startswith("PAIRED")]
    paired_dataset = CustomSubset(dataset, idx_paired)
    print(f"Len paired dataset: {len(paired_dataset)}")
    return paired_dataset


def create_render_dataset(root_dir,train_mode="char", imsize=224, font_name=""):

    if train_mode == "char":
        paired_transform = render_transform = create_paired_transform_char(imsize)
    else:
        paired_transform = render_transform = create_paired_transform(imsize)    
    dataset = FontImageFolder(root_dir, render_transform=render_transform, paired_transform=paired_transform)
    idx_render = [idx for idx, (p, t) in enumerate(tqdm(dataset.data)) if font_name in p and not os.path.basename(p).startswith("PAIRED")]
    render_dataset = CustomSubset(dataset, idx_render)
    print(f"Len render dataset: {len(render_dataset)}")

    return render_dataset


def create_hn_query_dataset(root_dir,train_mode="char", imsize=224,hn_query_list=[]):

    if train_mode == "char":
        paired_transform = render_transform = create_paired_transform_char(imsize)
    else:
        paired_transform = render_transform = create_paired_transform(imsize) 
    dataset = FontImageFolder(root_dir, render_transform=render_transform, paired_transform=paired_transform)

    unique_hn_query_list = (set(hn_query_list))

    idx_hn_query = [idx for idx, (p, t) in enumerate(tqdm(dataset.data)) if p in unique_hn_query_list]
    
    hn_query_dataset = CustomSubset(dataset, idx_hn_query)
    print(f"Len hn query dataset: {len(hn_query_dataset)}")
    
    return hn_query_dataset

