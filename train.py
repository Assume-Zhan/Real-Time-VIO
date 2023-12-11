import sys
sys.path.append('util')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pose_estimation.model import PoseNet
from util.data_process import get_data_info
from util.dataset_raw import ImageSequenceDataset, SortedRandomBatchSampler

from tqdm import tqdm

def lossFunction(prediction, correction):
    angle_loss = torch.nn.functional.mse_loss(prediction[:,:3], correction[:,:3])
    translation_loss = torch.nn.functional.mse_loss(prediction[:,3:], correction[:,3:])
    loss = (100 * angle_loss + translation_loss)
    return loss

if __name__ == '__main__':
    
    # Load Data with dataloader
    overlap = 0
    sample_times = 1
    folder_list = ['04']
    seq_len_range = [5, 7]
    df = get_data_info(seq_len_range, overlap, sample_times=sample_times,
                       data_path='./datasets/KITTI/training/04/*.png',
                       gt_path='./datasets/KITTI/pose_GT/04.npy')
    
    # Customized Dataset, Sampler
    n_workers = 1
    resize_mode = 'rescale'
    new_size = (150, 600)
    img_mean = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    dataset = ImageSequenceDataset(df, resize_mode, new_size, img_mean)
    sorted_sampler = SortedRandomBatchSampler(df, batch_size=2, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=sorted_sampler, num_workers=n_workers)
    
    # Setup model
    model = PoseNet()
    model = model.cuda()
    
    model.train()
    
    # Setup Optimizer
    optimizer = optim.SGD(model.parameters(), lr = 0.003)
    
    print('################# Start training #################')
    
    # Pass data
    for i in range(10):
        loss_all = 0
        for _, x, y in tqdm(dataloader):

            # y = y.view(y.size(0) * y.size(1), -1)
            
            x = torch.squeeze(x).cuda()
            y = torch.squeeze(y).cuda()

            # First size: batch size
            # Second size: sequence length
            # Third size: channel
            # Fourth size: height
            # Fifth size: width
            
            optimizer.zero_grad()

            # Normalize the input
            x = x / 255.0

            # forward + backward + optimize
            prediction = model(x)
            
            # compute loss here
            loss = lossFunction(prediction, y)
            loss_all += loss.item()
            
            # back prop
            loss.backward()
            optimizer.step()

        print('Epoch: {}, Loss: {}'.format(i, loss_all))
        
    print('################# Finish training #################')
        
        