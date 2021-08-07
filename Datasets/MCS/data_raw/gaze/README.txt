# Gaze Data Format

## Training Data

Training gaze data are contained in the following two files: "microwave_fixations.csv" and "clock_fixations.csv", whose columns are as follows:
- "subjectnum": subject id
- "TRIAL_INDEX": trial index of a subject
- "catcue": target object in the search task, either microwave or clock
- "condition": whether the target object exists in the image
- "expected": the expected/correct button number in a trial
- "im_h": image height
- "im_w": image width
- "searcharray": image name in the COCO2014 dataset
- "TRIAL_FIXATION_TOTAL": number of fixations in a trial
- "CURRENT_FIX_INDEX": fixation index in a trial
- "CURRENT_FIX_X": x-coordinate of the fixation
- "CURRENT_FIX_Y": y-coordinate of the fixation
- "CURRENT_FIX_DURATION": duration of the fixation
- "LAST_BUTTON_PRESSED": pressed button number by the subject in a trial

Notes:
1. During experiments, images are placed with its original size at the center of a 1280x800 display. 
2. Fixations are recorded in the coordinate of the display.
3. A trial is a correct trial where the subject correctly answered the question regarding whether a target object is present, if "LAST_BUTTON_PRESSED"=="expected".



## Testing Data

Testing gaze data are contained in "clock_and_microwave_fixations.csv", whose columns are as follows:
- "RECORDING_SESSION_LABEL": the subject number and target object, e.g., "s10c" means subject 10 and the target object is clock; while "s10m" means subject 10 and the target object is microwave.
- "TRIAL_INDEX": trial index of a subject
- "TRIAL_FIXATION_TOTAL": number of fixations in a trial
- "searcharray": image name in the COCO2014 dataset
- "target_present": whether the target object exists in the image. 1 is target-present, 0 is target-absent.
- "CURRENT_FIX_INDEX": fixation index in a trial
- "CURRENT_FIX_X": x-coordinate of the fixation
- "CURRENT_FIX_Y": y-coordinate of the fixation
- "CURRENT_FIX_DURATION": duration of the fixation
- "RESPONSE": response of the subject in a trial. 6 means correct answer; 7 means false positive; 5 means false negative.


Notes:
1. During experiments, images are resized and padded to fit a 1680x1050 display. 
2. Fixations are recorded in the coordinate of the display.
