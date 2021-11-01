import os
import glob
import json
import numpy as np
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import time

_DEFAULT_CROP_DIMS = False
if _DEFAULT_CROP_DIMS:
  _RESIZED_CROP_DIM = 224
  _CROP_DIM = 224
  _RESIZE_DIM = 256
else:
  _RESIZED_CROP_DIM = 64
  _CROP_DIM = 64
  _RESIZE_DIM = 86
  
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


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
    dataset_idx_lookup["val"] += list(val_idxs)
    dataset_idx_lookup["train"] += list(train_idxs)

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


class TinyImagenetDataset(Dataset):
  def __init__(
      self,
      root,
      dataset_key,
      val_split=None,
      split_idxs_root=None,
      load_previous_splits=True
    ):
    self.root = os.path.join(root, "TinyImageNet/data/")
    self.dataset_key = dataset_key
    if not os.path.exists(self.root):
      print("ERROR! Expects TinyImageNet data at <root>/TinyImageNet/data!")
    
    # Label lookup
    self.lookup = self._create_label_lookup()

    # Setup transforms and paths
    self._setup_transforms()
    
    # Set dataset path
    if dataset_key == "test":
      self.dataset_key = "test"
      self.dataset_path = os.path.join(self.root, "val")
    else:
      self.dataset_path = os.path.join(self.root, "train")
    
    self._setup_dataset_paths(
      val_split, 
      split_idxs_root, 
      load_previous_splits,
    )
    
  def _create_label_lookup(self):
    path = os.path.join(self.root, "wnids.txt")
    with open(path, "r") as infile:
      target_keys = [ele.strip() for ele in infile]

    lookup = {}
    path = os.path.join(self.root, "words.txt")
    with open(path, "r") as infile:
      key_i = 0
      for line in infile:
        key, name = line.strip().split("\t")
        if key not in target_keys:
          continue
        lookup[key] = (key_i, name)
        key_i += 1
    return lookup

  def _setup_dataset_paths(
      self, 
      val_split, 
      split_idxs_root,
      load_previous_splits
    ):
    # Load paths
    self.img_paths = glob.glob(f"{self.dataset_path}/*/*")

    # Compute num samples in dataset
    dataset_len = len(self.img_paths)

    if self.dataset_key != "test" and val_split is not None:
      # Set indices save/load path
      val_percent = int(val_split * 100)
      basename = f"{val_percent}-{100-val_percent}_val_split.json"
      idx_filepath = os.path.join(split_idxs_root, basename)
      print("idx_filepath: ", idx_filepath)
      # Check load indices
      if load_previous_splits and os.path.exists(idx_filepath):
        print(f"Loading previous splits from {idx_filepath}")
        with open(idx_filepath, "r") as infile:
          loaded_idxs = json.load(infile)
        dataset_idxs = loaded_idxs[self.dataset_key]
        print("Complete.")
      # Save idxs
      else:
        split_idxs_dict = split_dev_set(self.img_paths, val_split)
        dataset_idxs = split_idxs_dict[self.dataset_key]

        print(f"Saving split idxs to {idx_filepath}...")
        save_idxs = {
            key: [int(ele) for ele in vals]
            for key, vals in split_idxs_dict.items()
        }
        with open(idx_filepath, "w") as outfile:
          json.dump(save_idxs, outfile)
        print("Complete")

      self.img_paths = np.array(self.img_paths)[dataset_idxs]
    n_samples = len(self.img_paths)
    print(f"Loaded {self.dataset_key} with {n_samples} samples...")
  
  def _setup_transforms(self):
    if self.dataset_key == "train":
      xform_list = [
        T.RandomResizedCrop(_RESIZED_CROP_DIM), 
        T.RandomHorizontalFlip(p=0.5),
      ]
    else:
      xform_list = [
        T.Resize(_RESIZE_DIM), 
        T.CenterCrop(_CROP_DIM),
      ]

    xform_list += [
        T.ToTensor(),
        EnforceShape(),
        T.Normalize(mean=_MEAN, std=_STD)
    ]

    self.transforms = T.Compose(xform_list)

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    # Load img path
    path = self.img_paths[idx]

    # Load image
    img = Image.open(path)

    # Apply image transforms
    img = self.transforms(img)

    # Get label and y
    key = path.split(os.path.sep)[-2]
    y, label = self.lookup[key]
    
    return img, y  #, label
  
  
def create_datasets(
    data_root, 
    val_split, 
    split_idxs_root, 
    *args, 
    **kwargs
  ):
  dataset_dict = {}
  for dataset_key in ["train", "val", "test"]:
    print(f"Loading {dataset_key} data...")
    if dataset_key == "test":
      dataset_i = TinyImagenetDataset(data_root, dataset_key)
    else:
      dataset_i = TinyImagenetDataset(
        data_root, 
        dataset_key,
        val_split, 
        split_idxs_root,
      )
    
    dataset_dict[dataset_key] = dataset_i
  
  print(f"Complete.")
  return dataset_dict
