# Make a copick project
import os
import shutil

import copick

from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
from collections import defaultdict

from tqdm import tqdm
import numpy as np

from monai.networks.nets import UNet
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric

from torchinfo import summary
import mlflow.pytorch


from train.trainer import *
from czii_helper.helper import *

# -- config
cfg = dotdict(
    my_num_samples = 16, 
    train_batch_size = 1, 
    val_batch_size = 1, 
)

# -- 

config_blob = """{
    "name": "czii_cryoet_mlchallenge_2024",
    "description": "2024 CZII CryoET ML Challenge training data.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "apo-ferritin",
            "is_particle": true,
            "pdb_id": "4V1W",
            "label": 1,
            "color": [  0, 117, 220, 128],
            "radius": 60,
            "map_threshold": 0.0418
        },
        {
            "name": "beta-galactosidase",
            "is_particle": true,
            "pdb_id": "6X1Q",
            "label": 3,
            "color": [ 76,   0,  92, 128],
            "radius": 90,
            "map_threshold": 0.0578
        },
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "6EK0",
            "label": 4,
            "color": [  0,  92,  49, 128],
            "radius": 150,
            "map_threshold": 0.0374
        },
        {
            "name": "thyroglobulin",
            "is_particle": true,
            "pdb_id": "6SCJ",
            "label": 5,
            "color": [ 43, 206,  72, 128],
            "radius": 130,
            "map_threshold": 0.0278
        },
        {
            "name": "virus-like-particle",
            "is_particle": true,
            "label": 6,
            "color": [255, 204, 153, 128],
            "radius": 135,
            "map_threshold": 0.201
        },
        {
            "name": "membrane",
            "is_particle": false,
            "label": 8,
            "color": [100, 100, 100, 128]
        },
        {
            "name": "background",
            "is_particle": false,
            "label": 9,
            "color": [10, 150, 200, 128]
        }
    ],

    "overlay_root": "./working/overlay",

    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "/media/kyotaro/ubuntu_volume_1/Dataset/kaggle/czii-cryo-et-object-identification/train/static"
}"""

if __name__ == '__main__':
    input_dir = '/media/kyotaro/ubuntu_volume_1/Dataset/kaggle/czii-cryo-et-object-identification'
    output_dir = './working'
    os.makedirs(output_dir,exist_ok=True)

    copick_config_path = f"{output_dir}/copick.config"
    output_overlay = f"{output_dir}/overlay"

    with open(copick_config_path, "w") as f:
        f.write(config_blob)
        
    # Update the overlay
    # Define source and destination directories
    source_dir = f'{input_dir}/train/overlay'
    destination_dir = f'{output_dir}/overlay'

    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding subdirectories in the destination
        relative_path = os.path.relpath(root, source_dir)
        target_dir = os.path.join(destination_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy and rename each file
        for file in files:
            if file.startswith("curation_0_"):
                new_filename = file
            else:
                new_filename = f"curation_0_{file}"
            
            # Define full paths for the source and destination files
            source_file = os.path.join(root, file)
            destination_file = os.path.join(target_dir, new_filename)
            
            # Copy the file with the new name
            shutil.copy2(source_file, destination_file)
            # print(f"Copied {source_file} to {destination_file}")


    # -- get copick root
    root = copick.from_file(copick_config_path)

    # -- config for copick
    copick_user_name = "copickUtils"
    copick_segmentation_name = "paintedPicks"
    voxel_size = 10
    tomo_type = "denoised"


    # -- generate multi-calss segmentation masks from picks, and save them to the copick overlay directory once
    generate_masks=False
    if not os.path.exists(destination_dir+'/ExperimentRuns/TS_5_4/Segmentations'):
        generate_masks = True
    else:
        generate_masks = False

    if generate_masks:
        target_objects = defaultdict(dict)
        for object in root.pickable_objects:
            if object.is_particle:
                target_objects[object.name]['label'] = object.label
                target_objects[object.name]['radius'] = object.radius


        for run in tqdm(root.runs):
            tomo = run.get_voxel_spacing(10)
            tomo = tomo.get_tomogram(tomo_type).numpy()
            target = np.zeros(tomo.shape, dtype=np.uint8)
            for pickable_object in root.pickable_objects:
                pick = run.get_picks(object_name=pickable_object.name, user_id="curation")
                if len(pick):  
                    target = segmentation_from_picks.from_picks(pick[0], 
                                                                target, 
                                                                target_objects[pickable_object.name]['radius'] * 0.8,
                                                                target_objects[pickable_object.name]['label']
                                                                )
            write.segmentation(run, target, copick_user_name, name=copick_segmentation_name) 

    # -- get tomograms and their segmentation mask arrays
    data_dicts = []
    for run in tqdm(root.runs):
        tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type).numpy()
        segmentation = run.get_segmentations(name=copick_segmentation_name, user_id=copick_user_name, voxel_size=voxel_size, is_multilabel=True)[0].numpy()
        data_dicts.append({"image": tomogram, "label": segmentation})
    
    #set up dataloaders
    train_files, val_files = data_dicts[:5], data_dicts[5: 7]
    print(f'train {len(train_files)}samples / val {len(val_files)}samples')

    train_loader, val_loader, train_ds, val_ds \
          = make_train_val_dataloaders(train_files, val_files, cfg)

# -- define model (todo)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # Create UNet, DiceLoss and Adam optimizer
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(root.pickable_objects)+1,
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1,
    ).to(device)

    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr)
    #loss_function = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)  # softmax=True for multiclass
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  # softmax=True for multiclass
    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)  # must use onehot for multiclass
    recall_metric = ConfusionMatrixMetric(include_background=False, metric_name="recall", reduction="None")
# -- 


    mlflow.end_run()
    mlflow.set_experiment('training 3D U-Net model for the cryoET ML Challenge')
    epochs = 50

    with mlflow.start_run():
        params = {
            "epochs": epochs,
            "learning_rate": lr,
            "loss_function": loss_function.__class__.__name__,
            "metric_function": recall_metric.__class__.__name__,
            "optimizer": "Adam",
        }
        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        post_pred = AsDiscrete(argmax=True, to_onehot=len(root.pickable_objects)+1)
        post_label = AsDiscrete(to_onehot=len(root.pickable_objects)+1)

        train(
            train_ds, val_ds, 
            train_loader, val_loader, 
            model, loss_function, dice_metric, optimizer, epochs,
            device, post_pred, post_label,
            )

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model")