


#FDM is Fixation Density Maps.

#from pyedfread import edf

def get_subjectlist(search_pattern):
    #Returns a tuple that will store information on the EDF files 
    #that are going to be used in the analysis. This includes filesname, 
    #subject and run indices for each single EDF file.
    #
    #It will simply loop over all the files found recursively in the 
    #project_folder and guess the subject and run indices from the path.
    #
    #This method will most probably not work for others, therefore it is 
    #recommended that you generate the subjectlist tuple on your own and 
    #continue to continue with the analysis.
    #
    #Example:
    #search_pattern = "/mnt/data/project_FPSA_FearGen/data/**/data.edf"
    #eyepy.get_subjectlist(sp)
    #[('/Users/onat/Documents/Experiments/NoS/data/sub001/run001/eye/data.edf',
    #{'run': '001', 'subject': '001'}),
    
    import glob
    import re
    def get_subject():
        m = re.search('[(?<=sub)(?<=s)]\d{1,3}' , edf_path)
        return m.group()
    def get_run():
        #detects a 1 to 3 digits following either run, phase, r or p
        m = re.search('[(?<=run)(?<=phase)(?<=r)(?<=p)]\d{1,3}' , edf_path)
        return m.group()
    
    subject_list = [];
    for edf_path in glob.iglob(search_pattern,recursive=True):
        subject_list += [(edf_path,{'subject': get_subject(), 'run': get_run()})]
    return subject_list

def get_fixmat(filelist):
    #FILELIST is a list of EDF files that will be used for this project.
    
    #Call pyedfread and concat different data frames.
    with open(filelist) as fo:
        for count, name in enumerate(fo):
            print("Line {}: {}".format(count, name))
        #print("Line {}: {}".format(cnt, line))
        #samples, events, messages = edf.pread(f)

def bla():
    print(3)

#def get_FDM()

#def plot_FDM()

#def pattern_similarity()

#def classification()
    #we will want to classify participants (not caring about conditions). 
    #or conditions 
