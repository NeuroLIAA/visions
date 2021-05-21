# Data extraction

For subject `X` we have two files: `X.edf` and `X.mat`. The first file contains all the eye tracking information and the second one contains data about the subject, the image shown in each trial and their subjective response, among other stuff.

Firstly, we need to create the ASCII file, `X.asc`, from `X.edf`. We can do it with the following command:

```sh
wine edf2asc.exe *.edf
```

We need to create `X_info.mat`, a file that contains all the preprocessed information from the two sources (we extract features that we will be using a lot in the futures and we uniform some data stored using different conventions).

## How to create `X_info.mat` (the preferred way)
We firstly create `X_todo.mat`, a file that contains the information of `X.asc` in a format suitable for MATLAB. 

Then we create `X_info.mat` by taking that information, process it and merge it with data from `X_todo.mat`. Then we expand it with more features.

```sh
matlab create_info_from_raw_data() % this will create _info for all files in folder
```

## How to create `X_info.mat` if our `X.edf` is corrupt

It is important to know that `X.mat` has most of the eye tracking information contained in `X.edf` but with less precision, and it is intended to be used as backup. Therefore we can recover eye tracking data in the event of losing `X.edf`.

In this case, we will create the `X_info.mat` directly, without creating `X_todo.mat` first. 

```sh
matlab create_info_only_from_mat(<filename>)
```
