# Deep-Learning-with-Differential-Privacy

<img src="https://drive.google.com/uc?export=view&id=15hgLkUiSFYEvug29IE8FYL5hTa8q7MgL" style="width: 100px; max-width: 100%; height: auto"/>

### Objective

This project serves as an introduction to reading ML literature, and then applying this knowledge to deep learning and differential privacy concerns. The goal of this project is to understand deep learning models and how to protect the privacy of individual’s data. Different algorithmic techniques for learning will be implemented on medical image datasets and an analysis of privacy costs within the framework of differential privacy will be completed to evaluate the merits and room for improvement of different techniques. This project deliverable will be a research paper summarizing the results found throughout the fall and winter semester.

### Team Members

- Nicole Streltsov ([@NicoleStrel](https://github.com/NicoleStrel))
- Ritvik Jayanthi ([@RitvikJayanthi](https://github.com/Ritvik123487))
- Alec Dong ([@AlecDong](https://github.com/AlecDong))
- Ria Upreti ([@ria-upreti](https://github.com/ria-upreti))
- Akriti Sharma ([@Akriti-Sharma1](https://github.com/Akriti-Sharma1))
- Daniel Montero ([@danielfmontero](https://github.com/danielfmontero))
- Bolade Amoussou ([@cdw18](https://github.com/cdw18))
- Mikhael Orteza ([@xPreliator](https://github.com/xPreliator))

### Contents 

- Datasets:
  - /chest-data/: gzip numpy array files, from [Chest Pneumonia X-ray images dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
  - /knee-data/: gzip numpy array files, from [Knee Osteoarthritis X-ray images dataset](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity?select=auto_test)
- Techniques: 
  - /DP-SGD/: Differential Privacy in Stochiastic Gradient Descent, from the paper [Abadi et al.](https://arxiv.org/pdf/1607.00133.pdf)
- Python Scripts: 
  - load_dataset_into_pickle.py: reads directory of images, transforms the data into numpy arrays, applies data sugmentation and saves into gzip pickle files. 
  - visualize_dataset.py: reads directory of images to create a scatter plot of image size and label distribution. 
