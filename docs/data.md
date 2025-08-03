We have currently only open-sourced one scene from the dataset, as it takes time to anonymize all the data.

**1. Download ParkOcc dataset data [Baidu](https://pan.baidu.com/s/1Skh6c1fEACbiubBeAQXU8w?pwd=497a) or [Kaggle](https://www.kaggle.com/datasets/sheepsky/parkocc/data)（
mini dataset link: https://pan.baidu.com/s/19hNOV7JRp3xPFkzorH7_aw?pwd=7wa2   code: 7wa2 
）. Folder structure:**
```
ParkOcc
├── data/
│   ├── data_train/
│   │   ├── calib
│   │   ├── sequences
│   │   ├── semantic
│   │   ├── velodyne
```

**2. Put the generated train and val pickle files in data.**


**Folder structure:**
```
ParkOcc
├── data/
│   ├── data_train/
│   │   ├── calib
│   │   ├── sequences
│   │   ├── semantic
│   │   ├── velodyne
│   ├── parkocc_infos_train.pkl
│   ├── parkocc_infos_val.pkl

```
