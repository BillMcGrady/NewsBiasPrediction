# NewsBiasPrediction

This repository provides a reference implementation of news bias prediction model as described in the paper:<br>
> Encoding Social Information with Graph Convolutional Networks forPolitical Perspective Detection in News Media.<br>
> Chang Li, Dan Goldwasser.<br>
> Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.<br>
> https://www.aclweb.org/anthology/P19-1247/ <Insert paper link>

### Basic Usage

#### Data and Saved Models

Download the dataset from https://purdue0-my.sharepoint.com/:f:/g/personal/li1873_purdue_edu/EneD2GUyRCdLr6DWzfUuMysBLStW_zOGs6ht03HAq1IBIA?e=Tlf6pR. Put the 'data' folder in the root directory.

#### Example
Run: 

With SkipThought as text model
``python train.py -d newsbias_random --epochs 200 -lr 1e-2 -tm SkipThought``

With HLSTM as text model
``python train.py -d newsbias_random --epochs 50 -lr 1e-3 -tm HLSTM``

This program will load data from the 'data/' folder, construct (and train) embedding neural nets. We used 3-fold cross validation to get the average accuracy. Trained models will be stored under the folder 'saved_models/'. Note that following standard practice, we only release the URLs for the news articles we used (data/article_metadata.xml). Please contact us if more details are needed.

