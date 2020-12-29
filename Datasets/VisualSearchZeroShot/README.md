# Finding any Waldo with zero-shot invariant and efficient visual search Dataset
Visual search in natural images dataset which corresponds to the paper [Finding any Waldo with zero-shot invariant and efficient visual search](https://www.nature.com/articles/s41467-018-06217-x) (Zhang et. al, 2018). The repository containing the model and the complete datasets can be found [here](https://github.com/kreimanlab/VisualSearchZeroShot).

## Files descriptions
The **psy** folder contains all data gathered from human subjects, including the script and data files used in the experiment. In particular,
* **/ProcessScanpath_naturaldesign** includes all fixations from human subjects in MAT files, ordered by trial number.
* **HumanExp_NaturalDesign.m** describes the human experiment design.
* **naturaldesign.mat** indicates which image corresponds to each trial (480 trials in total).
* **naturaldesign_seq.mat** indicates which array index from naturaldesign.mat it'll be used for that specific trial (i.e. it selects the image to be displayed). The column number indicates the trial number.

The folders containing the images and targets used were excluded, and they can be found [here](https://drive.google.com/drive/folders/1-M-dEuom-UaqiYo3qULxI1-2rMnff4_O?usp=sharing). The images inside the **gt** folder indicate where the target is in the image and it is used to know if the human subject or the computational model has found it.
Note that the files inside the **target** folder are not exactly the target to be found in each displayed image, since they're not cropped directly from them.

The raw data of the experiment can be found in the folder **subjects_naturaldesign**, downloading *Part 1* from the Datasets section inside the VisualSearchZeroShot repository.
