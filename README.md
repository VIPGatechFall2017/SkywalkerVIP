# Support Vector Machine (SVM)
![alt text](https://amitranga.files.wordpress.com/2014/03/image44.png)

# Overview
Support vector machine, or SVM, is a type of supervised learning algorithm used for classification, regression, and outliers detection. This poject uses the SVM algorithm to classify ultrasound images of a right forearm muscle, and maps these images to which finger is being bent. We collected a total of 40 minutes of data of each group members moving each finger successively with a metronome. The video file is then converted into over 3500 png images, and trained these images using our data.txt file. 



# Collecting Data
[![Data Collection](https://img.youtube.com/vi/F-FhXAFbLvs/0.jpg)](https://www.youtube.com/watch?v=F-FhXAFbLvs&feature=youtu.be "ultrasound video")

Each group member sat down and collected data 5 minutes per one sitting. A 5 minute video file is converted into over 3500 png image files. Each of these images has a data.txt file that maps and identifies which finger is being bent. 

![alt text](https://i.imgur.com/sepKeoR.png)

These data files maps each png image with a vector with a value of 99 for each finger being bent. For the purpose of data processing, we've convereted these numbers into binary values of 1s and 0s, and then converted them again to a base-2 binary, so we can distinguish each finger with a unique value. 

# Prerequisites and Dependencies
You will need the following packages to run the code. To install the packages used in this project, run the following command.
```
pip install numpy, scipy scikit-learn, pandas
```






# Running the code

To test and run this code, you will need the following:
  * data.txt  //TODO link data.txt
  * corresponding png images that matches the data.txt //TODO link images 
  * svm.py


#

# Usage

Install `sklearn` and `numpy`. `cd` to the `svm` directory and run `python arrhythmia.py`

Dataset taken from [UCI arrhythmia dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/)

Read the comments for some details about the code
