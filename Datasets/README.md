# Datasets
Directory containing every dataset to be used. New datasets can be added by following the format and criteria described below.
## Preprocessing
For allowing a correct comparison, subjects' scanpaths are processed following these steps:
* Targets are regarded as having been found if the area covered by a given fixation (determined by an approximation of the size of the fovea) lands on the bounding box of the target.
* Consecutive fixations whose distance between each other is less than the estimated size of the fovea are lumped together.
* Scanpaths are cut as soon as the target is found.
* Search images where the initial fixation lands on the target, as well as scanpaths of length one, are excluded.
* Target templates, as well as their bounding boxes and COCO's category, are provided for all images.

The scripts that perform these changes can be found inside the ```data_raw``` folder of each dataset.

## Format
### Dataset's metadata
Two main files, located in the root directory of each dataset, provide the necessary metadata to perform the task: ```dataset_info.json``` and ```trials_properties.json```. The first one contains the general characteristics of the dataset (such as the paths to the search images, targets and participants’ scanpaths directories):
```json
    {
    "dataset_name": "[string] Dataset's name",
    "number_of_images": "[int] Total number of search images",
    "image_height": "[int] Height of the search images",
    "image_width": "[int] Width of the search images",
    "max_scanpath_length": "[int] Max. scanpath length allowed",
    "receptive_size": "[int, int] Fovea's size",
    "mean_target_size": "[int, int] Mean target size",
    "images_dir": "[string] Search images' path",
    "targets_dir": "[string] Targets' templates path",
    "scanpaths_dir": "[string] Participants' scanpaths path"
    }
```

```trials_properties.json```, on the other hand, specifies all the necessary information for running the models in each image:
```json
    [
        {
        "image": "[string] Search image filename",
        "target": "[string] Target template filename",
        "dataset": "[string] Dataset's name",
        "target_matched_row": "[int] Target's upper left row in the search image",
        "target_matched_column": "[int] Target's upper left column in the search image",
        "target_height": "[int] Target's height",
        "target_width": "[int] Target's width",
        "image_height": "[int] Search image's height",
        "image_width": "[int] Search image's width",
        "initial_fixation_row": "[int] Row of the initial fixation",
        "initial_fixation_column": "[int] Column of the initial fixation",
        "target_object": "[string] COCO's category of the target"
        },
        ...
    ]
```

### Subjects' scanpaths
Subjects are assigned a unique ID and the file which contains their scanpaths is named as ```subjID_scanpaths.json``` inside ```scanpaths_dir``` (specified in ```dataset_info.json```). Each individual scanpath is described as follows:

```json
    {
    "image_name":{
            "subject": "[string] Participant's ID",
            "dataset": "[string] Dataset's name",
            "image_height": "[int] Search image's height",
            "image_width": "[int] Search image's width",
            "screen_height": "[int] Screen's resolution height",
            "screen_width": "[int] Screen's resolution width",
            "receptive_height": "[int] Height, in pixels, of the fovea's size",
            "receptive_width": "[int] Width, in pixels, of the fovea's size",
            "target_found": "[bool] True if the target was found",
            "target_bbox": "[array[int]] Target's bounding box in the search image",
            "X": "[array[float]] Scanpath's column coordinates",
            "Y": "[array[float]] Scanpath's row coordinates",
            "T": "[array[float]] Scanpath's time, in ms",
            "target_object": "[string] COCO's category of the target",
            "max_fixations": "[int] Scanpath's length upper bound"
        }
        ...
    }
```

Note this format is equal to the one employed by visual search models to save their results (see [```Models```](../Models)), so a concordance between subjects and computational models is achieved.
