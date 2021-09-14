import splitfolders # pip install split-folders

splitfolders.ratio("images", output="cancer_data_split", seed=1337, ratio=(.8, .1, .1), group_prefix=None)  # default values
