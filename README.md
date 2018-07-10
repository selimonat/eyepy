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

# Example: Sanity Checks


<img src="https://github.com/selimonat/eyepy/blob/master/doc/sanity_check01.png" height="400">
<img src="https://github.com/selimonat/eyepy/blob/master/doc/sanity_check02.png" height="400">






