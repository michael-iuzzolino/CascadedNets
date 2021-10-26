import os
import glob
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from PIL import Image
from datasets.ImagenetClassLabels import ImagenetLabels
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import time

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def split_dataset(path_df, val_split=0.1, test_split=0.1, split_idxs_root=""):
  if not split_idxs_root:
    raise AssertionError("split_idxs_root not specified!")
    
  basename = f"{val_split}-{test_split}_val_test_split.json"
  split_path = os.path.join(split_idxs_root, basename)
  if os.path.exists(split_path):
    print(f"Loading splic locs from {split_path}...")
    with open(split_path, "r") as infile:
      split_locs = json.load(infile)
  else:
    split_locs = defaultdict(list)
    for class_lbl, class_df in path_df.groupby("class_lbl"):
      n_samples = len(class_df)
      n_test = int(n_samples * test_split)
      n_dev = n_samples - n_test
      n_val = int(n_dev * val_split)

      test_locs = list(class_df.iloc[:n_test].index)
      val_locs = list(class_df.iloc[n_test:n_test+n_val].index)
      train_locs = list(class_df.iloc[n_test+n_val:].index)

      split_locs["test"] += test_locs
      split_locs["val"] += val_locs
      split_locs["train"] += train_locs
      
    # Save
    print(f"Saving split locs to {split_path}...")
    with open(split_path, "w") as outfile:
      json.dump(split_locs, outfile)
  return split_locs


def subset_path_df(path_df, params):
  target_classes = params.get("target_classes", [])
  if target_classes:
    subset_dfs = []
    for target_class in target_classes:
      df_i = path_df[path_df.class_lbl.str.contains(target_class)]
      subset_dfs.append(df_i)
    path_df = pd.concat(subset_dfs)
  
  n_subset_classes = path_df.class_lbl.unique().shape[0]
  max_classes = params.get("max_classes", 0)
  if max_classes and max_classes < n_subset_classes:
    cls_counts = Counter(path_df.class_lbl)
    count_df = pd.DataFrame({
      "class_lbl": list(cls_counts.keys()), 
      "count": list(cls_counts.values())
    })
    count_df = count_df.sort_values("count", ascending=False)
    selected_classes = list(count_df.iloc[:max_classes].class_lbl)
    path_df = path_df[path_df.class_lbl.isin(selected_classes)]
  
  # Set class labels
  class_labels = np.sort(path_df.class_lbl.unique())
  lookup = {lbl: i for i, lbl in enumerate(class_labels)}
  path_df["y"] = [lookup[class_lbl] for class_lbl in path_df.class_lbl]
  
  return path_df

def build_path_df(root, subdir='ImageNet2012/ILSVRC/Data/CLS-LOC/train'):
  # Label handler
  label_handler = ImagenetLabels()
  
  # Set path
  dataset_path = os.path.join(root, subdir)
  
  # Class dirs
  class_dirs = glob.glob(f'{dataset_path}/*')

  df_dict = defaultdict(list)
  for class_dir in class_dirs:
    class_id = os.path.basename(class_dir)
    class_lbl = label_handler.lookup_lbl(class_id)

    img_paths = glob.glob(f"{class_dir}/*")
    for img_path in img_paths:
      df_dict["class_lbl"].append(class_lbl)
      df_dict["class_id"].append(class_id)
      df_dict["path"].append(img_path)
  path_df = pd.DataFrame(df_dict)
  return path_df

    
def split_dev_set(img_paths, val_split):
  class_paths = defaultdict(list)
  for idx, im_path in enumerate(img_paths):
    class_id = os.path.basename(os.path.dirname(im_path))
    class_paths[class_id].append((idx, im_path))

  dataset_idx_lookup = defaultdict(list)
  for class_key, vals in class_paths.items():
    vals = np.array(vals)
    idxs = vals[:, 0].astype(np.int)
    paths = vals[:, 1]

    n_val_samples = int(len(idxs) * val_split)
    val_idxs = np.random.choice(idxs, n_val_samples, replace=False)
    train_idxs = list(set(idxs).difference(set(val_idxs)))
    dataset_idx_lookup['val'] += list(val_idxs)
    dataset_idx_lookup['train'] += list(train_idxs)

  dataset_idx_lookup = {
      key: np.array(val)
      for key, val in dataset_idx_lookup.items()
  }
  return dataset_idx_lookup


class EnforceShape:
  """ Catches and converts grayscale and RGBA --> RGB """
  def __call__(self, x):
    if x.shape[0] == 1:
      x = x.repeat(3, 1, 1)
    elif x.shape[0] > 3:
      x = x[:3]
    return x


class ImagenetDataset(Dataset):
  def __init__(self, path_df, dataset_key):
    self.path_df = path_df
    self.dataset_key = dataset_key

    # Setup transforms and paths
    self._setup_transforms()

  def _setup_transforms(self):
    if self.dataset_key == 'train':
      xform_list = [T.RandomResizedCrop(224), T.RandomHorizontalFlip(p=0.5)]
    else:
      xform_list = [T.Resize(256), T.CenterCrop(224)]

    xform_list += [
        T.ToTensor(),
        EnforceShape(),
        T.Normalize(mean=_MEAN, std=_STD)
    ]

    self.transforms = T.Compose(xform_list)

  def __len__(self):
    return len(self.path_df)

  def __getitem__(self, idx):
    # Load df
    df_i = self.path_df.iloc[idx]
               
    # Load img path
    path = df_i.get("path")

    # Load image
    img = Image.open(path)

    # Apply image transforms
    img = self.transforms(img)

    # Get label and y
    label = df_i.get("class_lbl")
    y = df_i.get("y")

    return img, y  #, label


def create_datasets(path_df, val_split, test_split, split_idxs_root):
  # Label handler
  label_handler = ImagenetLabels()

  # Make sure split idx root exists
  if not os.path.exists(split_idxs_root):
    print(f"Creating {split_idxs_root}...")
    os.makedirs(split_idxs_root)
  
  # Split data
  split_locs = split_dataset(path_df, val_split, test_split, split_idxs_root)

  # Train
  print("Loading train data...")
  train_df = path_df.loc[split_locs["train"]]
  train_dataset = ImagenetDataset(train_df, 'train')
  print(f"{len(train_dataset):,} train examples loaded.")

  # Validation
  print("Loading validation data...")
  val_df = path_df.loc[split_locs["val"]]
  val_dataset = ImagenetDataset(val_df, 'val')
  print(f"{len(val_dataset):,} train examples loaded.")

  # Test
  print("Loading test data...")
  test_df = path_df.loc[split_locs["test"]]
  test_dataset = ImagenetDataset(test_df, 'test')
  print(f"{len(test_dataset):,} test examples loaded.")

  # Package
  dataset_dict = {
      "train": train_dataset,
      "val": val_dataset,
      "test": test_dataset,
  }
  print(f"Complete.")
  return dataset_dict