# Training Multimedia Event Extraction With Generated Images and Captions

Table of Contents
=================
  * [Overview](#overview)
  * [Requirements](#requirements)
  * [Data](#data)
  * [Quickstart](#quickstart)
  * [Citation](#citation)

## Overview
The code for paper [Training Multimedia Event Extraction With Generated Images and Captions](https://arxiv.org/pdf/2306.08966.pdf).


## Requirements

You can install the environment using `requirements.txt` for each component.

```pip
pip install -r requirements.txt
```

## Data

Visual event extraction data can be downloaded from [imSitu](http://imsitu.org/).
Textual event extraction data can be downloaded from [ACE](https://www.ldc.upenn.edu/).
We preprcoess data following [UniCL](https://github.com/jianliu-ml/Multimedia-EE) and [JMEE](https://github.com/lx865712528/EMNLP2018-JMEE/tree/master).


## Quickstart

### Generating data

(1) Generating Images
```
python Data/image_generator.py 
```
(2) Generating Captions
```
python Data/image_captioner.py 
```

### Training

(0) Data Preprocessing
```
python Data/object_detector.py 
```

Please specify the data paths, checpoint paths, output paths, and round (`decouple`, `round1`, and  `CKPT` `CKPT_OUTPUT`, `Event_Output_Path`) in the code. 

(1) Event Extraction
```
python Training/event_train.py 
```

(2) Argument Extraction
```
python Training/T_arg_train.py 
python Training/V_arg_train.py 
```

(3) Multimodal
```
python Training/mmee.py 
```


## Citation

```
@inproceedings{10.1145/3581783.3612526,
author = {Du, Zilin and Li, Yunxin and Guo, Xu and Sun, Yidan and Li, Boyang},
title = {Training Multimedia Event Extraction With Generated Images and Captions},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3612526},
doi = {10.1145/3581783.3612526},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {5504–5513},
numpages = {10},
keywords = {data augmentation, multi-modal learning, event extraction, cross-modality generation},
location = {<conf-loc>, <city>Ottawa ON</city>, <country>Canada</country>, </conf-loc>},
series = {MM '23}
}
```
