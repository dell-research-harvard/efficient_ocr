from pytorch_metric_learning.utils import common_functions as c_f
from torch.utils.data.sampler import Sampler
import torch
from collections import defaultdict

###Some utility functions
###We have a dict label_to_indices - dict[label] = list of indices
###We also have a dict idx:path
###So, we can make a dict label_to_paths - dict[label] = list of paths
###Now, if Path contains "PAIRED", it is paired data and otherwsie synthetic data. We want a label_to_paired_indices and label_to_unpaired_indices

def get_label_to_paths(dataset):
    labels = dataset.targets
  
    labels_to_indices = c_f.get_labels_to_indices(labels)    
    idx_to_path=dataset.path_dict
    labels_to_paths=defaultdict(list)
    for label in labels_to_indices:
        for idx in labels_to_indices[label]:
            labels_to_paths[label].append(idx_to_path[idx])
    
    return labels_to_paths


def get_label_to_paired_unpaired_indices(dataset):
    labels = set(dataset.targets)
    targets_dict = dataset.subsetted_targets_dict 
    idx_to_path=dataset.subsetted_path_dict

    ###We have mappings idx:labels (targets) and idx:path
    ###First we want a mapping label:paths. 
    labels_to_paths=defaultdict(list)
    for idx in idx_to_path:
        labels_to_paths[targets_dict[idx]].append(idx_to_path[idx][0])

    ###Now, we want to split the paths into paired and unpaired
    

    
    ## we now have a map between idx and path and label to paths. We now want to create a map between label to idx.
    ##idx_to_path is a dict idx:path. So, we need to invert it
    path_to_idx=defaultdict(list)
    for idx in idx_to_path:
        path_to_idx[idx_to_path[idx][0]].append(idx)

    label_to_paired_indices=defaultdict(list)
    label_to_unpaired_indices=defaultdict(list)
    for label in labels_to_paths:
        for path in labels_to_paths[label]:
            if "PAIRED-" in path:
                label_to_paired_indices[label].extend(path_to_idx[path])
            else:
                label_to_unpaired_indices[label].extend(path_to_idx[path]) 
    

    
    return label_to_paired_indices, label_to_unpaired_indices


class NoReplacementMPerClassSampler(Sampler):

    def __init__(self, dataset, m, batch_size, num_passes):
        labels = dataset.targets
        assert not batch_size is None, "Batch size is None!"
        if isinstance(labels, torch.Tensor): labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size)
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        print(f"Length of single pass: {self.length_of_single_pass}")
        self.dataset_len = int(self.length_of_single_pass * num_passes) # int(math.ceil(len(dataset) / batch_size)) * batch_size
        print(f"Dataset len: {self.dataset_len}")
        assert self.dataset_len >= self.batch_size
        # assert self.length_of_single_pass >= self.batch_size, f"m * (number of unique labels ({len(self.labels)}) must be >= batch_size"
        assert self.batch_size % self.m_per_class == 0, "m_per_class must divide batch_size without any remainder"
        self.dataset_len -= self.dataset_len % self.batch_size

    def __len__(self):
        return self.dataset_len

    def __iter__(self):

        idx_list = [0] * self.dataset_len
        i = 0; j = 0
        num_batches = self.calculate_num_batches()
        num_classes_per_batch = self.batch_size // self.m_per_class
        c_f.NUMPY_RANDOM.shuffle(self.labels)

        indices_remaining_dict = {}
        for label in self.labels:
            indices_remaining_dict[label] = set(self.labels_to_indices[label])

        for _ in range(num_batches):
            curr_label_set = self.labels[j : j + num_classes_per_batch]
            j += num_classes_per_batch
            assert len(curr_label_set) == num_classes_per_batch, f"{j}, {len(self.labels)}"
            if j + num_classes_per_batch >= len(self.labels):
                # print(f"All unique labels/classes batched, {len(self.labels)}; restarting...")
                c_f.NUMPY_RANDOM.shuffle(self.labels)
                j = 0
            for label in curr_label_set:
                t = list(indices_remaining_dict[label])
                if len(t) == 0:
                    randchoice = c_f.safe_random_choice(self.labels_to_indices[label], size=self.m_per_class).tolist()
                elif len(t) < self.m_per_class:
                    randchoice = t + c_f.safe_random_choice(self.labels_to_indices[label], size=self.m_per_class-len(t)).tolist()
                else:
                    randchoice = c_f.safe_random_choice(t, size=self.m_per_class).tolist()
                indices_remaining_dict[label] -= set(randchoice)
                idx_list[i : i + self.m_per_class] = randchoice
                i += self.m_per_class
        
        notseen_count = 0
        for k in indices_remaining_dict.keys():
            notseen_count += len(indices_remaining_dict[k])
        print(f"Samples not seen: {notseen_count}")

        return iter(idx_list)

    def calculate_num_batches(self):
        print(f"Dataset len: {self.dataset_len}")
        print(f"Batch size: {self.batch_size}")
        assert self.batch_size <= self.dataset_len, "Batch size is larger than dataset!"
        return self.dataset_len // self.batch_size


class HardNegativeClassSamplerChar(Sampler):

    def __init__(self, 
            dataset, 
            classidx, 
            hardnegs, 
            hnset_per_batch=1, 
            m=4, 
            batch_size=128, 
            hns_set_size=8,
            num_passes=1
        ):

        labels = dataset.targets

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        self.labels = labels
        print(f"Number of hard negative sets: {len(hardnegs)}")

        self.all_labels_for_negs = []

        for hns in hardnegs:
            try:
                lab_neg_set = [classidx[str(ord(c))] for c in hns]
                assert len(lab_neg_set) == hns_set_size
                self.all_labels_for_negs.append(lab_neg_set)
            except KeyError:
                ##Split hard negs by |
                hns = hns.split("|")
                lab_neg_set = [classidx[c] for c in hns]
                assert len(lab_neg_set) == hns_set_size
                self.all_labels_for_negs.append(lab_neg_set)
        
        self.batch_size = batch_size
        self.m_per_class = m
        self.hnset_per_batch = hnset_per_batch

        self._sampler = NoReplacementMPerClassSampler(
            dataset=dataset, m=m, batch_size=batch_size, num_passes=num_passes
        )

    def __len__(self):
        return len(self._sampler)

    def __iter__(self):

        _idx_list = list(self._sampler.__iter__())
        c_f.NUMPY_RANDOM.shuffle(self.all_labels_for_negs)
        labels_to_indices = c_f.get_labels_to_indices(self.labels)
        all_hn_indices = []

        indices_remaining_dict = {}
        for label in self.labels:
            indices_remaining_dict[label] = set(labels_to_indices[label])

        for hn_labels_for_batch in self.all_labels_for_negs:
            hn_idx_for_batch = []
            for label in hn_labels_for_batch:
                t = list(indices_remaining_dict[label])
                if len(t) == 0:
                    t = labels_to_indices[label].tolist()
                if len(t) != 0: # label/underlying char is in eval set...
                    if len(t) < self.m_per_class:
                        randchoice = t + c_f.safe_random_choice(labels_to_indices[label], size=self.m_per_class-len(t)).tolist()
                    else:
                        randchoice = c_f.safe_random_choice(t, size=self.m_per_class).tolist()
                    assert len(randchoice) == self.m_per_class
                    indices_remaining_dict[label] -= set(randchoice)
                    hn_idx_for_batch.extend(randchoice)
            all_hn_indices.append(hn_idx_for_batch)

        
        for hni in all_hn_indices:
            ridx = c_f.NUMPY_RANDOM.choice(range(0, len(_idx_list), self.batch_size))
            _idx_list[ridx:ridx] = hni

        print(f"Number of samples in epoch (hard negatives): {len(_idx_list)}")
        
        return iter(_idx_list) 


class AllHNSamplerSplitBatchesPairRender(Sampler):
    """Further split batch into digital and synthetic fonts"""

    def __init__(self, 
        dataset, 
        classidx, 
        hardnegs, 
        m=4, 
        batch_size=128, 
        hns_set_size=8,
        num_passes=1 #Not used in this sampler, here for compatibility
        ):

        assert batch_size % (m*hns_set_size) == 0
        labels = dataset.targets
        self.dataset=dataset

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        self.labels = labels
        print(f"Number of hard negative sets: {len(hardnegs)}")

        self.all_labels_for_paired_negs=[]
        self.all_labels_for_multi_char_negs=[]

        multi_char_hardnegs = hardnegs
        print(f"Number of multi char hard negative sets: {len(multi_char_hardnegs)}")

        
        for hns in multi_char_hardnegs:
            ##Split hard negs by |
            hns=hns.split("|")[:hns_set_size]
            lab_neg_set = [classidx[c] for c in hns]
            assert len(lab_neg_set) == hns_set_size
            self.all_labels_for_multi_char_negs.append(lab_neg_set)
            
        print("Total labels for multi-char negs" , len(self.all_labels_for_multi_char_negs))
        print("Total labels for multi-char negs - first index" , len(self.all_labels_for_multi_char_negs[0]))


        self.batch_size = batch_size
        self.m_per_class = m
        print("Check",int(self.m_per_class/2.0))
        self.hnset_per_batch = int(batch_size / (m*hns_set_size))
        self.hns_set_size=hns_set_size

        # self._sampler = NoReplacementMPerClassSampler(
        #     dataset=dataset, m=m, batch_size=batch_size, num_passes=num_passes)
        
    @property
    def nbatches(self):
        num_multi=len(self.all_labels_for_multi_char_negs)
        num_total=num_multi
        self.num_total=num_total*self.hns_set_size*self.m_per_class
        self.num_total_batches=round(self.num_total/self.batch_size)
        
        
        return  self.num_total_batches        

    def __len__(self):
        return len(self._sampler)

    def __iter__(self):


        labels_to_paired_indices, labels_to_unpaired_indices=get_label_to_paired_unpaired_indices(self.dataset)
        

        all_multi_char_hn_indices = []

        indices_remaining_dict_paired = {}
        indices_remaining_dict_unpaired = {}
        for label in self.labels:
            indices_remaining_dict_paired[label] = set(labels_to_paired_indices[label])
            indices_remaining_dict_unpaired[label] = set(labels_to_unpaired_indices[label])
            assert set(labels_to_paired_indices[label]).intersection(set(labels_to_unpaired_indices[label])) == set()

        def process_hn_labels(hn_labels, all_hn_indices):
            for hn_labels_for_batch in hn_labels:
                hn_idx_for_batch = []
                for label in hn_labels_for_batch:
                    t_paired = list(indices_remaining_dict_paired[label])
                    t_unpaired = list(indices_remaining_dict_unpaired[label])

                    if len(t_paired) == 0:
                        t_paired = labels_to_paired_indices[label]
                    if len(t_unpaired) == 0:
                        t_unpaired = labels_to_unpaired_indices[label]

                    num_paired = (self.m_per_class // 2)
                    num_unpaired = (self.m_per_class // 2)

                    # Select paired examples
                    if len(t_paired)==0:
                        randchoice_paired=[]
                    else:
                        randchoice_paired = c_f.safe_random_choice(t_paired, size=num_paired).tolist()
                    indices_remaining_dict_paired[label] -= set(randchoice_paired)

                    if len(t_paired)!=0:
                        assert len(randchoice_paired) == num_paired
                    # print("randchoice_paired", randchoice_paired)
                    hn_idx_for_batch.extend(randchoice_paired)


                    # Select unpaired examples
                    randchoice_unpaired = c_f.safe_random_choice(t_unpaired, size=int(self.m_per_class-len(randchoice_paired))).tolist()
                    indices_remaining_dict_unpaired[label] -= set(randchoice_unpaired)
                    assert len(randchoice_unpaired) == int(self.m_per_class-len(randchoice_paired)) 
                    # print("randchoice_unpaired", randchoice_unpaired)
                    if len(randchoice_paired)==0:
                        assert len(randchoice_unpaired) == int(self.m_per_class)
                    else:
                        assert len(randchoice_unpaired) == num_unpaired
                    hn_idx_for_batch.extend(randchoice_unpaired)

                    # print(len(hn_idx_for_batch), self.m_per_class)

                assert len(hn_idx_for_batch) == int(self.m_per_class * self.hns_set_size)
                all_hn_indices.append(hn_idx_for_batch)


        process_hn_labels(self.all_labels_for_multi_char_negs, all_multi_char_hn_indices)

        
        c_f.NUMPY_RANDOM.shuffle(all_multi_char_hn_indices)

        print("Total number of multi_char_indices", len(all_multi_char_hn_indices))
        print("Length of a set in multi_char_indices", len(all_multi_char_hn_indices[10]))

        ###Now, we want every batch to have a mix of single and multi char hard negatives. 
        ###Each batch will have self.hnset_per_batch number of hard negative sets. Half of them will be single char, half will be multi char
        ###We want to sample multi_char_hn_sets without replacement 
        ##But, We will sample single_char_hn_Sets WITH replacement
        ##By doing so, the total number of ids would double

        all_hn_indices=[]
        i=0
        while len(all_multi_char_hn_indices)>0:

            all_hn_indices.append(all_multi_char_hn_indices.pop())

            i+=1
        
        print("Total number of hard negative sets: ", len(all_hn_indices))
        print("Example hard negative set: ", all_hn_indices[0], all_hn_indices[1])
        print("Example Length of a set" , len(all_hn_indices[0]))
                


        _idx_list=[item for sublist in all_hn_indices for item in sublist]



        print(f"Number of samples in epoch (hard negatives): {len(_idx_list)}")
        ##Number of batches

        print("Number of batches", len(_idx_list)/self.batch_size)

        return iter(_idx_list)

