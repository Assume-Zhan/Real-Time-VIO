import os
import glob
import numpy as np
import pandas as pd
from helper import R_to_angle

# Given the poseGT [R|t], transform it to [theta_x, theta_y, theta_z, x, y, z]
# And store it as .npy file
# Sample reference:
# https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/bb43825e54b0d96d40bfe8b30013d5438f607908/preprocess.py#L47-L60
def create_pose_data(folder='../datasets/KITTI/pose_GT/'):
    
    # Specified whitch data to use
	info = {'04': [0, 270]}
 
    # Read from each sequence data
	for video in info.keys():
		fn = '{}{}.txt'.format(folder, video)

        # Open the file
		with open(fn) as f:
      
            # Read lines from the file
			lines = [line.split('\n')[0] for line in f.readlines()]

            # Transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
			poses = [ R_to_angle([float(value) for value in l.split(' ')]) for l in lines]
			
            # TODO: we need to calculate the difference here
			# Calculate the difference between each pose
			# poses = [[poses[i] - poses[i-1]] for i in range(1, len(poses))]
   
            # Save as .npy file
			poses = np.array(poses)
			base_fn = os.path.splitext(fn)[0]
			np.save(base_fn +'.npy', poses)
   
			# Print the shape of the data
			print('Video {}: shape={}'.format(video, poses.shape))

# Sample reference:
# https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/bb43825e54b0d96d40bfe8b30013d5438f607908/data_helper.py#L15-L78
def get_data_info(seq_len_range, 
                  overlap, 
                  data_path='../datasets/KITTI/training/image_0/*.png',
                  gt_path='../datasets/KITTI/pose_GT/04.npy',
                  sample_times=1,
                  shuffle=False, 
                  sort=True):
    
    X_path, Y = [], []
    X_len = []
    
    # Get the path for data and ground truth pose
    fpaths = glob.glob(data_path)
    poses = np.load(gt_path)
    fpaths.sort()
    
    # Fixed seq_len
    if seq_len_range[0] == seq_len_range[1]:
        if sample_times > 1:
            # Sample multiple times
            sample_interval = int(np.ceil(seq_len_range[0] / sample_times))
            start_frames = list(range(0, seq_len_range[0], sample_interval))
            print('Sample start from frame {}'.format(start_frames))
        else:
            start_frames = [0]

        for st in start_frames:
            seq_len = seq_len_range[0]
            n_frames = len(fpaths) - st
            jump = seq_len - overlap
            res = n_frames % seq_len
            if res != 0:
                n_frames = n_frames - res
            x_segs = [fpaths[i:i+seq_len] for i in range(st, n_frames, jump)]
            y_segs = [poses[i:i+seq_len] for i in range(st, n_frames, jump)]
            Y += y_segs
            X_path += x_segs
            X_len += [len(xs) for xs in x_segs]
            
    # Random segment to sequences with diff lengths
    else:
        assert(overlap < min(seq_len_range))
        n_frames = len(fpaths)
        
        # calculate the length of each segment
        min_len, max_len = seq_len_range[0], seq_len_range[1]
        
        # Sample multiple times
        for _ in range(sample_times):
            start = 0
            while True:
                
                # Randomly sample a length
                n = np.random.randint(min_len, max_len + 1)
                
                if start + n < n_frames:
                    
                    # Sample data by the length
                    x_seg = fpaths[start:start+n] 
                    X_path.append(x_seg)
                    
                    # Sample ground truth pose by the length
                    Y.append(poses[start:start+n])
     
                else:
                    # Last few frames are thrown away
                    print('Last %d frames is not used' %(start+n-n_frames))
                    break
                start += n - overlap
                X_len.append(len(x_seg))
    
    # Convert to pandas dataframes
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
    
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
        
    # Sort dataframe by seq_len
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
        
    return df
   
# Test the above functions
if __name__ == '__main__':
	create_pose_data()

	overlap = 1
	sample_times = 1
	folder_list = ['04']
	seq_len_range = [5, 7]
	df = get_data_info(seq_len_range, overlap, sample_times=sample_times)