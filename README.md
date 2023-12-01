# ML-Term-Project
Machine Learning Term Projject

## Background for Visual Odometry

| Topics | Description | Reference Links | Other |
|--------------|-------------|--------------------------|-------|
| Dataset | KITTI dataset | [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) | - |

## KITTI Dataset Download

ground truth : for all 00 to 10 folders are included(in .txt)

training dataset : for linux (make sure to go to the KITTI folder)
for now it only download folder 04
```bash 
cd datasets/KITTI
./downloader.sh
```

for windows for now you must download it manually and follow the structure.

check out downloader.sh to see which file to download for each folder (for example folder 00 correspond to 2011_10_03_drive_0027) 

- datasets/
  - KITTI/
    - training/
        - 00/
            - (bunch of png files from images_03)/
        - 01/
        - 02/
        - 03/
        - 04/ 




