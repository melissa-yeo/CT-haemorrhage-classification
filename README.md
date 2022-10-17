# CT-haemorrhage-classification

Code for intracranial haemorrhage detection on CT head imaging.\
The CNN-RNN model is based on the implementation by [darraghdog](https://github.com/darraghdog/rsna).

### Files
1. Preparing training dataset (Kaggle):
     - raw dataset available at: https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data
     - split_kaggle_dataset.py
     - preprocess_kaggle_dataset.py 
2. Preparing test dataset (CQ500):
     - raw dataset available at: http://headctstudy.qure.ai/dataset
     - preprocess_CQ500_dataset.py
     - optional: reads_interrater_analysis.py to evaluate ratings
3. Training/ testing CNN: 
     - CNN.py
4. Training/ testing RNN:
     - LSTM.py
5. Evaluation:
     - eval_report.py
     - gradcam.py

### Hardware
This research was undertaken using the LIEF HPC-GPGPU Facility hosted at the University of Melbourne. \
This Facility was established with the assistance of LIEF Grant LE170100200. \
Two Intel Xeon E5-2650 CPUs (24 cores)\
Four NVIDIA P100 GPUs 

### Software
Refer to envi.yml\
Python 3.6\
PyTorch 1.4.0

