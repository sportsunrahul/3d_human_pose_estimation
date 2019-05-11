## Installation
The code was tested with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v1.0.1. After installing Anaconda and Pytorch:

1. Clone the repo or use the source code provided:

    ~~~
    git clone https://github.com/sportsunrahul/3d_human_pose_estimation.git
    ~~~


2. Install dependencies (opencv, and progressbar):

    ~~~
    conda install --channel https://conda.anaconda.org/menpo opencv
    conda install --channel https://conda.anaconda.org/auto progress
    ~~~

3. Optionally, install tensorboard for visializing training. 

    ~~~
    pip install tensorflow
    ~~~

## Demo
- Download our pre-trained [model](https://drive.google.com/file/d/1Ud2x79tdKX1VHk17dkPHqk_S7WIA4tXp/view?usp=sharing) and move it to `models`.
- Run `python demo.py --demo /path/to/image/or/image/folder [--gpus -1] [--load_model /path/to/model]`. 

`--gpus -1` is for CPU mode. 
We provide example images in `images/`. For testing your own image, it is important that the person should be at the center of the image and most of the body parts should be within the image. 

## Benchmark Testing
To test our model on Human3.6 dataset run 

~~~
python main.py --exp_id test --task human3d --dataset fusion_3d --load_model ../models/fusion_3d_var.pth --test --full_test
~~~

The expected results are 64.19mm.

## Training
- Prepare the training data:
  - Download images from [MPII dataset](http://human-pose.mpi-inf.mpg.de/#download) and their [annotation](https://onedrive.live.com/?authkey=%21AKqtqKs162Z5W7g&id=56B9F9C97F261712%2110696&cid=56B9F9C97F261712) in json format (`train.json` and `val.json`) (from [Xiao et al. ECCV2018](https://github.com/Microsoft/human-pose-estimation.pytorch)).
  - Download [Human3.6M ECCV challenge dataset](http://vision.imar.ro/human3.6m/challenge_open.php).
  - Download [meta data](https://www.dropbox.com/sh/uouev0a1ao84ofd/AADzZChEX3BdM5INGlbe74Pma/hm36_eccv_challenge?dl=0&subfolder_nav_tracking=1) (2D bounding box) of the Human3.6 dataset (from [Sun et al. ECCV 2018](https://github.com/JimmySuen/integral-human-pose)). 
  - Place the data (or create symlinks) to make the data folder like: 
  
  ```
  ${POSE_ROOT}
  |-- data
  `-- |-- mpii
      `-- |-- annot
          |   |-- train.json
          |   |-- valid.json
          `-- images
              |-- 000001163.jpg
              |-- 000003072.jpg
  `-- |-- h36m
      `-- |-- ECCV18_Challenge
          |   |-- Train
          |   |-- Val
          `-- msra_cache
              `-- |-- HM36_eccv_challenge_Train_cache
                  |   |-- HM36_eccv_challenge_Train_w288xh384_keypoint_jnt_bbox_db.pkl
                  `-- HM36_eccv_challenge_Val_cache
                      |-- HM36_eccv_challenge_Val_w288xh384_keypoint_jnt_bbox_db.pkl
  ```

- Stage1: Train 2D pose only.

```
python main.py --exp_id mpii
```

- Stage2: Train on 2D and 3D data without geometry loss (drop LR at 45 epochs). 

```
python main.py --exp_id fusion_3d --task human3d --dataset fusion_3d --ratio_3d 1 --weight_3d 0.1 --load_model ../exp/mpii/model_last.pth --num_epoch 60 --lr_step 45
```

- Stage3: Train with geometry loss.

```
python main.py --exp_id fusion_3d_var --task human3d --dataset fusion_3d --ratio_3d 1 --weight_3d 0.1 --weight_var 0.01 --load_model ../models/fusion_3d.pth  --num_epoch 10 --lr 1e-4
```
