import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from helper import normalize_angle_delta

# Image Sequence Dataset
# Prepare the training data with some data augmentation
# Reference:
# https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/bb43825e54b0d96d40bfe8b30013d5438f607908/data_helper.py#L179-L240
class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
        transform_ops.append(transforms.ToTensor())
        
        self.minus_point_5 = minus_point_5
        
        # Setup Transformer and Normalizer
        self.transformer = transforms.Compose(transform_ops)
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        
        # Setup data
        self.data_info = info_dataframe
        
        # Sequence length for each mini batch
        self.seq_len_list = list(self.data_info.seq_len)
        
        # Setup data
        self.image_arr = np.asarray(self.data_info.image_path)
        
        # Setup ground truth
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def __getitem__(self, index):
        raw_groundtruth = np.hsplit(self.groundtruth_arr[index], np.array([6]))	
        groundtruth_sequence = raw_groundtruth[0]
        groundtruth_rotation = raw_groundtruth[1][0].reshape((3, 3)).T # opposite rotation of the first frame
        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)
        # groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0:-1]  # get relative pose w.r.t. previois frame 

        groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0] # get relative pose w.r.t. the first frame in the sequence 
		
        # print('Item before transform: ' + str(index) + '   ' + str(groundtruth_sequence))

        # here we rotate the sequence relative to the first frame
        for gt_seq in groundtruth_sequence[1:]:
            location = torch.FloatTensor(groundtruth_rotation.dot(gt_seq[3:].numpy()))
            gt_seq[3:] = location[:]
            # print(location)

        # get relative pose w.r.t. previous frame
        groundtruth_sequence[2:] = groundtruth_sequence[2:] - groundtruth_sequence[1:-1]

		# here we consider cases when rotation angles over Y axis go through PI -PI discontinuity
        for gt_seq in groundtruth_sequence[1:]:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])

        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])
        
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)

# Sampler for sampling the batch with sorted sequence length
# Reference:
# https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/bb43825e54b0d96d40bfe8b30013d5438f607908/data_helper.py#L144-L176
class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        # Calculate len (num of batches, not num of samples)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        
        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len