# How to use the optical flow model?

## To access the KittiFlow dataset
The program expected the dataset in the following directory structure:
```
\optical_flow_model
    \datasets
        \KITTI
            \testing
            \training
```
where the `\testing` and `\training` files are downloaded from [here](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) (The 2 GB one)

After you put the dataset in this structure, the dataset can be accessed by:
(You don't have to program this)
```
dataset = dataset_kittiflow.KITTI(split='training')
```

## To train a model (both finetune or from scratch)
Use the command:
```
python train.py --name <desired name for the model> --stage kitti --validation kitti --restore_ckpt <the path to the checkpoint model> --gpus 0 --num_steps 1000 --batch_size 1 --lr 0.0001 --image_size 288 960 --wdec
ay 0.00001 --gamma=0.85 --mixed_precision --small
```
> Note that:
> - The above hyperparameters are provided from the original RAFT [repo](https://github.com/princeton-vl/RAFT), you can adjust it any way you want!
> - If you want to **train the model from scratch**, please remove the `--restore_ckpt` argument.

## To inference with your model
Use the command:
```
# If you want to inference the testing data of kittiflow dataset
python inference.py --data=kittiflow --model=<path to your model> --iters <number of iteration for the model to inference> --output_path=<desired output path for the flow images> --small

# If you want to inference your own images
python inference.py --data=<path to your images> --model=<path to your model> --iters <number of iteration for the model to inference> --output_path=<desired output path for the flow images> --small
```
> Note that:
> - Your own images should be in `.png` or `.jpg` format.
## To evaluate your model
This will output the following metrics of your model:
- **end-point error (epe)**
- **f1-score**
- **average inferencing fps**

Use the command:
```
python evaluate.py --model=<path to your model> --small --mixed_precision --iters <number of iteration for the model to inference>
```