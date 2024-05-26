## 3D Unet for segmentation

This project implements 2 different 3D Unets for 3d model segmentation.
One mode is a default implementation from monai, and one custom model created with pytorch.

Both models where trained on the Asoca-grand dataset


### Usages
Configs are in the configs.yaml file

The data was given in .nrrd files. Further usage of the models, needs a new glob/dataloader

training mode, **checkpoint_path** path to .ptr file from torch, **optional: Test** bool if run one cycle and stop.
````
python ./asoca-grand/train_segment_model.py <optional: checkpoint_path> <optional: Test (bool)>
````
