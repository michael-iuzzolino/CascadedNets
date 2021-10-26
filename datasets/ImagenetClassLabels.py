import os
import urllib
import json
import numpy as np

_JSON_URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


class ImagenetLabels:
  def __init__(self, save_path=None, json_url=None):
    if json_url is None:
      json_url = _JSON_URL

    self.json_url = json_url
    self.savepath = savepath if save_path is not None else '/tmp/imagenet_labels.json'

    self._download()
    self._load_labels()

  def _download(self):
    if os.path.exists(self.savepath):
      print(f"File aleady exists at {self.savepath}.")
    else:
      print(f"Downloading from {self.json_url}")
      urllib.request.urlretrieve(self.json_url, self.savepath)

  def _load_labels(self):
    print(f"Loading data from {self.savepath}")
    with open(self.savepath, 'r') as infile:
      data = json.load(infile)

    data = {int(key): val for key, val in data.items()}
    self.idx_to_class = {key: val[0] for key, val in data.items()}
    self.idx_to_label = {key: val[1] for key, val in data.items()}

    self.cls_to_idx = {val[0]: key for key, val in data.items()}
    self.label_to_idx = {val[1]: key for key, val in data.items()}

    self.cls_to_label = {val[0]: val[1] for key, val in data.items()}
    self.label_to_cls = {val: key for key, val in self.cls_to_label.items()}

    self.num_classes = len(data)
    print("Fin.")

  def sample_classes(self, num_classes=10, target_labels=[]):
    if len(target_labels):
      sampled_labels = []
      for target_label in target_labels:
        sampled_labels += [lbl 
                           for lbl in self.labels 
                           if target_label.lower() in lbl.lower()]

    else:
      sampled_labels = list(
          np.random.choice(self.labels, num_classes, replace=False))
    
    # Get classes from labels
    sampled_classes = [self.label_to_cls[ele] for ele in sampled_labels]
    return sampled_labels, sampled_classes

  @property
  def labels(self):
    return list(self.label_to_cls.keys())

  @property
  def classes(self):
    return list(self.cls_to_label.keys())
  
  def lookup_cls(self, label):
    return self.label_to_cls[label]
  
  def lookup_lbl(self, cls):
    return self.cls_to_label[cls]