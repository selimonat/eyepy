# EyePy
Python package for classification of eye-movement patterns using machine learning. 

EyePy is constantly growing to the extend I find time to code.

+  Import .EDF files into a Pandas DataFrame object using the Pyedfread toolbox written by Niklas Wilming.
+  Find and discard invalid fixation data.
+  Run simple sanity checks to ensure the quality of your data is acceptable for running classification algorithms
+  Run different supervised classification algorithms on the data to classify different participants.

## Credits
This package is coded by Selim Onat. 
Algorithms were initially developed in Matlab during Master thesis of Lea Kampermann (supervision: Selim Onat). 
Niklas Wilming has been regularly consulted for Python coding tipps, as well as data analysis topics.

In the following, use of the toolbox is illustrated step-by-step/

# Where to start

## Selecting files for the analysis

To import EDF files, EyePy will recursively search a path. 
It will deduce participants' ID and if a participant is recorded multiple times, it will also try to guess the run ID. These information will be stored in a tuple of size N, where N is the number of EDF files found.

One may want to pile all EDF files into a single folder, or keep them within a project folder hierarchy. 

### Example:
```
search_pattern = "/mnt/data/project_FPSA_FearGen/data/\*/p02/\*/data.edf" 
```

The above search_pattern would find all the following EDF files:

```
file_list = eyepy.get_filelist(search_pattern)

('/mnt/data/project_FPSA_FearGen/data/sub027/p02/eye/data.edf',
  {'run': 2, 'subject': 27}),
 ('/mnt/data/project_FPSA_FearGen/data/sub074/p02/eye/data.edf',
  {'run': 2, 'subject': 74}),
 ('/mnt/data/project_FPSA_FearGen/data/sub070/p02/eye/data.edf',
  {'run': 2, 'subject': 70}),
 ('/mnt/data/project_FPSA_FearGen/data/sub029/p02/eye/data.edf',
  {'run': 2, 'subject': 29}),
 ('/mnt/data/project_FPSA_FearGen/data/sub043/p02/eye/data.edf',
  {'run': 2, 'subject': 43}),
 ('/mnt/data/project_FPSA_FearGen/data/sub055/p02/eye/data.edf',
  {'run': 2, 'subject': 55}),
 ('/mnt/data/project_FPSA_FearGen/data/sub063/p02/eye/data.edf',
  {'run': 2, 'subject': 63}),...
```   
Note that the run and participant IDs are automatically generated from the path.

### Importing data

EyePy relies on Pandas DataFrame to store and manipulate data. 
All EDF files in file_list are stored as a DataFrame in df.
```
df = eyepy.get_df(fl)
```
df is a Pandas dataframe object that contains all the fixation data.
``` 
In [30]: df.head()
Out[30]: 
      end  fix        gavx        gavy  run   start  subject trial validation_result:         DISPLAY_COORDS  file  condition  oddball  ucs  py_trial_marker  weight
0   491.0  0.0  846.900024  619.299988    2   365.0       27     1                NaN  [550, 350, 1049, 849]   3.0     -135.0      0.0  0.0              1.0       1
1   693.0  1.0  737.900024  556.599976    2   534.0       27     1                NaN  [550, 350, 1049, 849]   3.0     -135.0      0.0  0.0              1.0       1
2  1038.0  2.0  706.200012  543.900024    2   718.0       27     1                NaN  [550, 350, 1049, 849]   3.0     -135.0      0.0  0.0              1.0       1
3  1214.0  3.0  844.200012  553.299988    2  1084.0       27     1                NaN  [550, 350, 1049, 849]   3.0     -135.0      0.0  0.0              1.0       1
4  1438.0  4.0  718.299988  539.799988    2  1259.0       27     1                NaN  [550, 350, 1049, 849]   3.0     -135.0      0.0  0.0              1.0       1
```

In this DataFrame, each row corresponds to one fixation data with attributes stored in the columns. 
The naming of columns follows closely the Pyedfread toolbox.

Importing also uses JobLib for caching data to disk.

### Sanity Checks

```
eyepy.sanity_checks(df)
```
provides information for assessing the quality of the data set.

(1) Distribution of the calibration error: 
The histogram below shows the distribution of calibration error, with a vertical line which indicates the generally accepted
maximum calibration error (.5 degrees) that can be tolerated in a study.
A printed message will suggest which subjects has to be discarded based on the above-mentionned criterium.
```
There are 8 participants ([55 40 56 69 75 77 48 46]) with a calibration error
that is more than     half a degree. You might consider
excluding them from the analysis.
````
(2) Trial counts per conditions and participants:
The second plot on the right is a two dimensional histogram that indicates the number of fixations per participant and condition.
It can be helpful for finding participants which were not attentive during the recordings. 
In this case, participant 11 seems to have a lot of fixation data missing.

<img src="https://github.com/selimonat/eyepy/blob/master/doc/sanity_check01.png" width="600">

(3) Space and time of fixations

Second figure features a scatter plot that shows x, y and time information of each fixation point. 
This can be useful to ensure the good quality of the dataset.
Different participants are color coded along the blue-purple color continuum. 
The histogram of fixations along the x and y coordinates shows the central tendency of the fixations to be located on the monitor.
Furthermore, the histogram of fixation ranks shows the expected decrease in higher ranks.

<img src="https://github.com/selimonat/eyepy/blob/master/doc/sanity_check02.png" width="600">

## Data Analysis

### Hiererchical Clustering of Fixation Maps

EyePy relies on the Group object created by the GroupBy method in DataFrame objects. 
For example, the following snippet computes a similarity matrix using the fixation maps of different subjects.
This returns a two dimensional, symmetric and non-negative numpy array.
```
sim_mat = eyepy.pattern_similarity(df.groupby('subject'))
```
This  similarity matrix can be used for constructing a dendrogram as shown below.
```
eyepy.dendrogram(s)
```
<img src="https://github.com/selimonat/eyepy/blob/master/doc/hierarchical_01.png" width="600">





