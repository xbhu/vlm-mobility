---
annotations_creators: []
language: en
license: bsd
task_categories:
- image-classification
- object-detection
task_ids: []
pretty_name: bdd100k-validation
tags:
- fiftyone
- image
- image-classification
- object-detection
dataset_summary: '



  ![image/png](dataset_preview.gif)



  This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 10000 samples.


  ## Installation


  If you haven''t already, install FiftyOne:


  ```bash

  pip install -U fiftyone

  ```


  ## Usage


  ```python

  import fiftyone as fo

  import fiftyone.utils.huggingface as fouh


  # Load the dataset

  # Note: other available arguments include ''split'', ''max_samples'', etc

  dataset = fouh.load_from_hub("dgural/bdd100k")


  # Launch the App

  session = fo.launch_app(dataset)

  ```

  '
---

# Dataset Card for bdd100k-validation


From one of the largest open source driving datasets, [BDD100k](https://www.vis.xyz/bdd100k/), is the BDD100K images dataset.
The dataset consists of every 10th second in the videos and contains a train, validation and test split.
It contains labels for object detection, weather, time of day, and scene of the driving!


![image/png](dataset_preview.gif)


This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 10000 samples.

## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

# Load the dataset
# Note: other available arguments include 'split', 'max_samples', etc
dataset = fouh.load_from_hub("dgural/bdd100k")

# Launch the App
session = fo.launch_app(dataset)
```


## Dataset Details

### Dataset Description


- **Curated by:** [ETH VIS Group](https://www.vis.xyz/)
- **Language(s) (NLP):** en
- **License:** bsd

### Dataset Sources [optional]


- **Repository:** [bdd100k](https://github.com/bdd100k/bdd100k)
- **Paper :** [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning](https://arxiv.org/abs/1805.04687)

## Uses

By downoading, you are agreeing to the [BDD100K License](https://doc.bdd100k.com/license.html#license). 
Copyright ©2018. The Regents of the University of California (Regents). All Rights Reserved.

THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 1/1/2021

Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement; and permission to use, copy, modify and distribute this software for commercial purposes (such rights not subject to transfer) to BDD and BAIR Commons members and their affiliates, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED “AS IS”. REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

### Direct Use

This dataset is great for self driving car applications, especially for dealing with many different weather conditions and different times of day!


## Dataset Structure
```
Name:        bdd100k-validation
Media type:  image
Num samples: 10000
Persistent:  True
Tags:        []
Sample fields:
    id:         fiftyone.core.fields.ObjectIdField
    filepath:   fiftyone.core.fields.StringField
    tags:       fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:   fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.ImageMetadata)
    weather:    fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
    timeofday:  fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
    scene:      fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
    detections: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
```
    



### Source Data

The dataset is sourced from [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning](https://arxiv.org/abs/1805.04687)


#### Who are the source data producers?

BDD100K is now managed by the [ETH VIS Group](https://www.vis.xyz/bdd100k/)


## Citation

**BibTeX:**

@misc{yu2020bdd100k,
      title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning}, 
      author={Fisher Yu and Haofeng Chen and Xin Wang and Wenqi Xian and Yingying Chen and Fangchen Liu and Vashisht Madhavan and Trevor Darrell},
      year={2020},
      eprint={1805.04687},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
