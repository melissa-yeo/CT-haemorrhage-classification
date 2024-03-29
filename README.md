# CT-haemorrhage-classification

Code for intracranial haemorrhage detection on CT head imaging.

### Files
1. Preparing training dataset (Kaggle):
     - raw dataset available at: https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data
     - run split_kaggle_dataset.py
     - run preprocess_kaggle_dataset.py 
2. Preparing test dataset (CQ500):
     - raw dataset available at: http://headctstudy.qure.ai/dataset
     - run preprocess_CQ500_dataset.py
     - optional: run reads_interrater_analysis.py to evaluate ratings
3. Training/ testing CNN: 
     - run CNN.py
4. Training/ testing RNN:
     - run LSTM.py
5. Evaluation:
     - run eval_report.py
     - run gradcam.py

### Hardware
Two Intel Xeon E5-2650 CPUs (24 cores).\
Four NVIDIA P100 GPUs.\
This research was undertaken using the LIEF HPC-GPGPU Facility hosted at the University of Melbourne. \
This Facility was established with the assistance of LIEF Grant LE170100200. 

### Software
Refer to envi.yml\
Python 3.6\
PyTorch 1.4.0

