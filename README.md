# Local Citation Recommendation by Graph Convolutional Networks with BERT Embeddings

This repository contains a PyTorch implementation of Graph Convolutional Networks (GCNs) with BERT Embeddings for local citation recommendation. The project aims to enhance citation recommendation systems by leveraging graph-based neural networks and contextual embeddings from BERT.

## Datasets

Datasets used in this project are available for download at [Google Drive](https://drive.google.com/drive/folders/11n4YVHgUPfzetJi-y5voFpmRIjiBM0lQ).

Available datasets for training:
* FullTextPeerRead
* ACL-200
* RefSeer
* arXiv

## Installation

Installation instructions will be available soon. Stay tuned!

## Usage
To parse a dataset, run:

```python dataset-parser.py "datasetname"```

To train the model with a specific dataset, run:

```python train.py --dataset_name="datasetname"```

Please refer to train.py for additional parameters and customization options.