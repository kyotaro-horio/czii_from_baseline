import torch

from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    AsDiscrete,  
    RandFlipd, 
    RandRotate90d, 
    NormalizeIntensityd,
    RandCropByLabelClassesd,
)

def make_train_val_dataloaders(train_files, val_files, config):
    
    my_num_samples      = config.batch_size
    train_batch_size    = 1
    val_batch_size      = 1

    # Non-random transforms to be cached
    non_random_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ])

    # Random transforms to be applied during training
    random_transforms = Compose([
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[config.patch_size, config.patch_size, config.patch_size],
            num_classes=7,
            num_samples=my_num_samples
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),    
    ])

    # Create the cached dataset with non-random transforms
    train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=1.0)

    # Wrap the cached dataset to apply random transforms during iteration
    train_ds = Dataset(data=train_ds, transform=random_transforms)

    # DataLoader remains the same
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # Validation transforms
    val_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[config.patch_size, config.patch_size, config.patch_size],
            num_classes=7,
            num_samples=my_num_samples,  # Use 1 to get a single, consistent crop per image
        ),
    ])

    # Create validation dataset
    val_ds = CacheDataset(data=val_files, transform=non_random_transforms, cache_rate=1.0)

    # Wrap the cached dataset to apply random transforms during iteration
    val_ds = Dataset(data=val_ds, transform=random_transforms)

    # Create validation DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,  # Ensure the data order remains consistent
    )

    return train_loader, val_loader, train_ds, val_ds