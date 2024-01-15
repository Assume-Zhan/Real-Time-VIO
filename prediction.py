# Do prediction
import os
import glob
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from pose_estimation.model import PoseNet
from util.data_process import get_data_info
from util.dataset_raw import ImageSequenceDataset, SortedRandomBatchSampler
from util.helper import eulerAnglesToRotationMatrix

from optical_flow_model.models.raft import RAFT
from optical_flow_model.utils.utils import InputPadder
import optical_flow_model.utils.dataset_kittiflow as dataset_kittiflow

def lossFunction(prediction, correction):
    angle_loss = torch.nn.functional.mse_loss(prediction[:,:3], correction[:,:3])
    translation_loss = torch.nn.functional.mse_loss(prediction[:,3:], correction[:,3:])
    loss = (100 * angle_loss + translation_loss)
    return loss

def prediction_1(model):

    overlap = 0
    sample_times = 1
    seq_len_range = [6, 6]
    dataset = get_data_info(seq_len_range, overlap, sample_times=sample_times,
                            data_path='./datasets/KITTI/opimage/*.png',
                            gt_path='./datasets/KITTI/pose_GT/04.npy')
    resize_mode = 'rescale'
    new_size = (150, 600)
    img_mean = (0.8633468176473288, 0.8739793531183159, 0.9322413316673415)
    dataset = ImageSequenceDataset(dataset, resize_mode, new_size, img_mean)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    answer = [[0.0]*6, ]
    st_t = time.time()

    # Start Predicting
    model.eval()

    for idx, (_, x, y) in enumerate(dataloader):

        # y = y.view(y.size(0) * y.size(1), -1)

        x = torch.squeeze(x).cuda()
        y = torch.squeeze(y).cuda()

        # First size: sequence length
        # Second size: channel
        # Third size: height
        # Fourth size: width

        # Remove the last image channel in x
        x = x[:, :-1, :, :]
        prediction = model.predict(x)

        prediction = prediction.data.cpu().numpy()

        # if idx == 0:
        for pose in prediction[0]:
            # use all predicted pose in the first prediction
            for i in range(len(pose)):
                # Convert predicted relative pose to absolute pose by adding last pose
                pose[i] += answer[-1][i]
            answer.append(pose.tolist())
            print("0 : Predicting {}th image sequence".format(idx))
        prediction = prediction[1:]

        # print('prediction.shape: ', prediction.shape)

        for predict_pose_seq in prediction:

            # predict_pose_seq[1:] = predict_pose_seq[1:] + predict_pose_seq[0:-1]
            ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0]) #eulerAnglesToRotationMatrix([answer[-1][1], answer[-1][0], answer[-1][2]])
            location = ang.dot(predict_pose_seq[-1][3:])
            predict_pose_seq[-1][3:] = location[:]

            last_pose = predict_pose_seq[-1]
            for i in range(len(last_pose)):
                last_pose[i] += answer[-1][i]
            # normalize angle to -Pi...Pi over y axis
            last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
            answer.append(last_pose.tolist())
            print("Predicting {}th image sequence".format(idx))

    print('len(answer): ', len(answer))
    # print('expect len: ', len(glob.glob('{}{}/*.png'.format(par.image_dir, test_video))))
    print('Predict use {} sec'.format(time.time() - st_t))


    # Save answer
    with open('{}/out_{}.txt'.format('./datasets/KITTI/result', '04'), 'w') as f:
        for pose in answer:
            if type(pose) == list:
                f.write(', '.join([str(p) for p in pose]))
            else:
                f.write(str(pose))
            f.write('\n')

if __name__ == '__main__':

    # Load model
    model = PoseNet().cuda()
    model.load_state_dict(torch.load('./pose_estimation/weights/weights_04.pth'))

    # Do prediction
    prediction_1(model)