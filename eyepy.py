


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
    
    import glob
    subject_list = None;
    for file in glob.iglob(search_pattern,recursive=True):                
        subject_list += [(file,{'subject': get_subject(), 'run': get_run()})]
    return subject_list

    def get_subject():
        return 1
    def get_run():
        return 2

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
