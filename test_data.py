import os
from glob import glob

import monai.bundle.utils
import numpy as np
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstd, EnsureTyped, CenterSpatialCropD, \
    RandSpatialCropD, RandScaleIntensityd, NormalizeIntensityd, RandShiftIntensityd, ToTensorD, ResizeD

# In[ ]:


# Load the configuration file
config_dict = {}
try:
    with open("configs.yaml", 'r') as stream:
        config_dict = monai.bundle.utils.yaml.load(stream, Loader=monai.bundle.utils.yaml.FullLoader)
except FileNotFoundError:
    print("Config file not found.")
    exit()

def get_paths(path_array):
    if isinstance(path_array, str):
        path_array = [path_array]
    allPaths = []
    for path in path_array:
        allPaths.append(sorted(glob(os.path.join(path, '*.nrrd'))))
    return [item for sublist in allPaths for item in sublist]


# In[ ]:


path_to_image = get_paths(config_dict['dataset']['image_path'])
path_to_masks = get_paths(config_dict['dataset']['mask_path'])

data = [{'image': image, 'mask': mask} for image, mask in zip(path_to_image, path_to_masks)]




train_transforms = Compose([
    LoadImageD(keys=["image", "mask"], reader="itkreader"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    EnsureTyped(keys=["image", "mask"]),
    CenterSpatialCropD(keys=["image", "mask"], roi_size=[400, 400, 200]),
    RandSpatialCropD(keys=["image", "mask"], roi_size=[350, 350, 180], random_size=False),
    ResizeD(keys=["image", "mask"], spatial_size=[256, 256, 112]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ToTensorD(keys=["image", "mask"]),
])

val_transform = Compose(
    [
        LoadImageD(keys=["image", "mask"], reader="itkreader"),
        EnsureChannelFirstd(keys=["image", "mask"]),
        EnsureTyped(keys=["image", "mask"]),
        CenterSpatialCropD(keys=["image", "mask"], roi_size=[350, 350, 180]),
        ResizeD(keys=["image", "mask"], spatial_size=[256, 256, 112]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensorD(keys=["image", "mask"]),
    ]
)

train_set = monai.data.CacheDataset(data[:int(len(data) * 0.8)], transform=train_transforms) if config_dict['dataset'][
    'cache_dataset'] else monai.data.Dataset(data[:int(len(data) * 0.8)], transform=train_transforms)

test_set = monai.data.CacheDataset(data[int(len(data) * 0.8):], transform=val_transform) if config_dict['dataset'][
    'cache_dataset'] else monai.data.Dataset(data[int(len(data) * 0.8):], transform=val_transform)


train_loader = monai.data.DataLoader(train_set, batch_size=config_dict['trainer']['batch_size'],
                                     num_workers=config_dict['trainer']['num_workers'], shuffle=True,
                                     collate_fn=monai.data.pad_list_data_collate)
test_loader = monai.data.DataLoader(test_set, batch_size=config_dict['trainer']['batch_size'],
                                    num_workers=config_dict['trainer']['num_workers'], shuffle=True,
                                    collate_fn=monai.data.pad_list_data_collate)

print(f"Train set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")

print(np.max(train_set[0]['image']), np.min(train_set[0]['image']))
print(np.max(train_set[0]['mask']), np.min(train_set[0]['mask']))

# In[ ]:

item = iter(train_loader).__next__()
print(item["image"].shape, item["mask"].shape)


