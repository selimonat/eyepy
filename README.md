# EyePy
Python package for classification of eye-movement patterns using machine learning. 
At the current version, it is specifically tailored to work with .EDF files of the SR-Research.

With this package you can
+  import EDF files of your project into a Pandas DataFrame object using the pyedfread toolbox.
+  run simple sanity checks to ensure that the quality of your data is acceptable for running machine learning classification algorithms
+  run different supervised classification algorithms on the data to classify different participants.

# Example: Select Files

To import your EDF files, EyePy will look recursively into your project folder and deduce run and participant ids. 
You may want to pile all your EDF files into one non-hierarchical folder, or keep your files within your project folder hierarchy.
As long as you provide the correct search_pattern, EyePy will detect the files.

For example, the following search search_pattern would find all these files:
search_pattern = "/mnt/data/project_FPSA_FearGen/data/**/p02/**/data.edf" 

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
   
Note that the run and participant IDs are automatically generated from the path.

# Example: Import data

EyePy relies on Pandas DataFrame to store and manipulate data. 
All EDF files in file_list are stored as a DataFrame in df.
df = eyepy.get_df(fl)
df.head(5)
  DISPLAY_COORDS     end  fix        gavx        gavy  run   start  subject  \
0            NaN   491.0  0.0  846.900024  619.299988    2   365.0       27   
1            NaN   693.0  1.0  737.900024  556.599976    2   534.0       27   
2            NaN  1038.0  2.0  706.200012  543.900024    2   718.0       27   
3            NaN  1214.0  3.0  844.200012  553.299988    2  1084.0       27   
4            NaN  1438.0  4.0  718.299988  539.799988    2  1259.0       27   

  trial validation_result:  file  condition  oddball  ucs  py_trial_marker  \
0     1                NaN   3.0     -135.0      0.0  0.0              1.0   
1     1                NaN   3.0     -135.0      0.0  0.0              1.0   
2     1                NaN   3.0     -135.0      0.0  0.0              1.0   
3     1                NaN   3.0     -135.0      0.0  0.0              1.0   
4     1                NaN   3.0     -135.0      0.0  0.0              1.0   

   weight  
0       1  
1       1  
2       1  
3       1  
4       1  

# Example: Sanity Checks

eyepy.sanity_checks(df)

provides crucial information for assessing the quality of the data set.

(1) Distribution of the calibration error: 

The figure shows which participants could be discarded based on their calibration error. It is common to use an average threshold of .5 degrees.

(2) Trial counts per conditions and participants:

This is important to realize whether there are any missing trials that could generate an unbalanced dataset.

<img src="https://github.com/selimonat/eyepy/blob/master/doc/sanity_check01.png" width="400">

(3) Space and time of fixations

scatter plots show x, y and time information of each fixation point, which is useful to 
ensure the good quality of the dataset.


<img src="https://github.com/selimonat/eyepy/blob/master/doc/sanity_check02.png" width="400">






