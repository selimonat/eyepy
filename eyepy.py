


#FDM is Fixation Density Maps.

#from pyedfread import edf

def get_filelist(search_pattern):
    '''
    Find files that will be used for the analysis and retrieves metadata about 
    participants and runs.
    
    Returns a tuple that stores information about EDF files. This includes 
    filename, subject and run indices for each single EDF file.
    
    It will simply loop over all files found recursively in the 
    project_folder and guess subject and run indices from the path. If this 
    returns unexpected values have a look at the nested get_{subject,run} 
    functions. It is strongly recommended that you create your own method to 
    generate a tuple for your project, in case this fails (which will most 
    likely happen).
    
    output: a list of tuples in the form [('filename', {"meta":'data'})]
    
    Example:
    search_pattern = "/mnt/data/project_FPSA_FearGen/data/**/data.edf"
    eyepy.get_filelist(sp)
    [('/Users/onat/Documents/Experiments/NoS/data/sub001/run001/eye/data.edf',
    {'run': '001', 'subject': '001'}),
    '''
    import glob
    import re
    def get_subject():
        m = re.search('((?<=sub)|(?<=s))\d{1,3}' , edf_path)
        return m.group()
    def get_run():
        #detects a 1 to 3 digits following either run, phase, r or p
        m = re.search('((?<=run)|(?<=phase)|(?<=r)|(?<=p))\d{1,3}' , edf_path)
        return m.group()
    
    filelist = [];
    for edf_path in glob.iglob(search_pattern,recursive=True):
        filelist += [(edf_path,{'subject': get_subject(), 'run': get_run()})]
    return filelist

def get_fixmat(filelist):
    '''
    Reads all EDF files and returns 3 dataframes based on pyedfread. 
    Uses metadata dictionary to label dataframes with subject and run information.
    
    See self.get_filelist to know more about filelist format.
    See pyedfread for more information on the output dataframes.
    '''
    from pyedfread import edf
    import pandas as pd
    import sys
    total_file = len(filelist)
    print("Receivied {} EDF files".format(total_file))
    #init 3 variables that will accumulate data frames as a list
    s = [None] * total_file
    e = [None] * total_file
    m = [None] * total_file
    #Call pyedfread and concat different data frames as lists.
    for i,file in enumerate(filelist):
        filename                  = file[0]        
        s[i], e[i], m[i] = edf.pread(filename,meta=file[1])
        #progress report
        sys.stdout.write('\r')        
        sys.stdout.write("Reading file {}\n".format(filename))                                
        sys.stdout.write("[%-20s] %d%%" % ('='*round((i+1)/total_file*20), round((i+1)/total_file*100)))
        sys.stdout.flush()
        
    #convert lists to data frame.
    samples = pd.concat(s)
    events  = pd.concat(e)
    messages= pd.concat(m)
    
    return samples, events, messages

def bla():
    print(3)

#def get_FDM()

#def plot_FDM()

#def pattern_similarity()

#def classification()
    #we will want to classify participants (not caring about conditions). 
    #or conditions 
