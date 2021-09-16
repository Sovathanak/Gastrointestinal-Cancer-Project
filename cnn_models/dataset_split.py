import splitfolders  # pip install split-folders

splitfolders.ratio("images", output="cancer_data_split", seed=1337, ratio=(.6, .2, .2), group_prefix=None)  # default values
