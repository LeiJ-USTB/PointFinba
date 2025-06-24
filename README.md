PointFinba: A Novel Fusion of Mamba and Finch for Advanced Point Cloud Learning
==

# Some need
```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Mamba install
pip install causal-conv1d
pip install mamba_ssm
```

# Pre-trained weights
Download the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/1F7Cf5BVWMqMGgdLDCZCithY2UomQozlJ?usp=sharing).
