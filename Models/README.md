# Models
Human visual search computational models to be tested. New models can be added by following the specifications described below.
## Methodology
For each trial, visual search is carried out until one of these two conditions is met: a) a fixation lands on the target's bounding box, or b) an upper saccade limit is reached.
## Input
Visual search models contain a *main.py* script in their root directory. This script, in turn, includes a method named ```main(dataset_name[, human_subject_id])```. The parameters are as follows:
* *dataset_name:* Name of the dataset on which to perform visual search. Its directory can be found under ```Datasets/dataset_name```, where two main files reside: ```dataset_info.json``` (dataset's metadata) and ```trials_properties.json``` (metadata of each trial, such as the search image file, target template and bounding box, etc.). The specifications of these files can be found in [```Datasets```](../Datasets).
* *human_subject_id*: Optional parameter, needed to perform Human Scanpath Prediction (see [```Metrics```](../Metrics)). If specified, the given subject's scanpaths are followed for each trial, where for each fixation *n* with *n* greater than zero (the initial fixation is fixation zero) the conditional priority map is saved under the name *‘fixation_n.csv’* in the same directory where the model’s results for that dataset reside (see Output). Once this is done for the whole scanpath, ```save_scanpath_prediction_metrics(subject_scanpath, image_name, output_path)``` in [*human_scanpath_prediction.py*](../Metrics/scripts/human_scanpath_prediction.py) is called. This method proceeds to compute the metrics for each fixation and stores the average.

Visual search models loop through ```Datasets/dataset_name/trials_properties.json```, performing visual search on each of the dataset's images, and store the results as described below.

## Output
Results of the visual search are stored in [```Results```](../Results), under the folder ```dataset_name/model_name``` in a file named ```Scanpaths.json```. The file's structure is as follows:
```json
    {
    "image_name":{
            "subject": "[string] Model's name",
            "dataset": "[string] Dataset's name",
            "image_height": "[int] Search image height",
            "image_width": "[int] Search image width",
            "receptive_height": "[int] Height, in pixels, of the fovea's size",
            "receptive_width": "[int] Width, in pixels, of the fovea's size",
            "target_found": "[bool] True if the target was found",
            "target_bbox": "[array[int]] Target's bounding box in the search image",
            "X": "[array[float]] Scanpath's column coordinates",
            "Y": "[array[float]] Scanpath's row coordinates",
            "target_object": "[string] COCO's category of the target",
            "max_fixations": "[int] Scanpath's length upper bound"
        },
    ...
    }
```

* If performing Human Scanpath Prediction, the conditional priority maps are stored in the same directory under the folder ```subjects_predictions/subject_id/probability_maps/image_name```.
