# DDGCN: A Dynamic Directed Graph Convolutional Network 


**Requirements:**

Cuda 10.0:
https://developer.nvidia.com/cuda-10.0-download-archive

Pytorch 1.2:
```conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch```

MMCV 4.4:
```pip install mmcv==0.4.4```

OpenCV:
```conda install -c conda-forge opencv```

**Application Usage**:

**1. OpenPose:**

 **1. a.  Action Recognition**

  - The user enters the following arguments for OpenPose action recognition:
   --rec   to select the action recognition option
   --pose  path to the OpenPose output directory. OpenPose output files are stored as JSON format for each frame
   --wi    video frames width for pose data normalization
   --he    video frames height for pose data normalization

  - Example of command-line script to run the OpenPose action recognition:
    Python PATH_TO_APP_ROOT/run_DDGCN_Open.py --rec --pose=PATH_TO_OPENPOSE_OUTPUT --wi=FRAMES_WIDTH --he=FRAMES_HEIGHT

  **1. b.  Visualization**

  - The user enters the following arguments for OpenPose skeleton visualization:
   --vis    to select the skeleton visualization option
   --pose   path to the OpenPose output directory. OpenPose output files are stored as JSON format for each frame
   --video  path to the video directory

  - Example of command-line script to run the OpenPose skeleton visualization:
    ```Python PATH_TO_APP_ROOT/run_DDGCN_Open.py --vis --pose=PATH_TO_OPENPOSE_OUTPUT --video=PATH_TO_VIDEO```

**2. NTU:**
 - The user enters the following arguments for NTU action recognition:
   --pose  path to the NTU skeleton sequence following its standards

 - Example of command-line script to run the NTU action recognition:
   ```Python PATH_TO_APP_ROOT/run_DDGCN_NTU.py --pose=PATH_TO_SKELETON_DATA```
  
**Some example scripts:**

```cd C:\DDGCN_running\DDGCN_Kinect_OpenPose```

```Python run_DDGCN.py --rec --pose=C:/DDGCN_running/DDGCN_Kinect_OpenPose/data/openpose_output --wi=640 --he=360```

```Python run_DDGCN.py --vis --pose=C:/DDGCN_running/DDGCN_Kinect_OpenPose/data/visualization/openpose_output --video=C:/DDGCN_running/DDGCN_Kinect_OpenPose/data/visualization/gulf.mp4```
