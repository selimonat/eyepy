


#FDM is Fixation Density Maps.
import numpy as np
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
        filelist += [(edf_path,{'subject': int(get_subject()), 'run': int(get_run())})]
    return filelist

def get_fixmat(filelist):
    '''
    Reads all EDF files with pyedfread and returns a dataframe. 
    Output dataframe merges all events, metadata and messages.
    Relies on metadata dictionary (see get_filelist) to label the dataframe with 
    subject and run information.
    
    See self.get_filelist to know more about filelist format.
    See pyedfread for more information on the output dataframes.
    '''
    from pyedfread import edf    
    import pandas as pd
    import sys
    total_file = len(filelist)
    print("Receivied {} EDF files".format(total_file))
    #init 3 variables that will accumulate data frames as a list
    e = [None] * total_file
    m = [None] * total_file
    #Call pyedfread and concat different data frames as lists.
    for i,file in enumerate(filelist):
        filename                  = file[0]        
        _, e[i], m[i] = edf.pread(filename,meta=file[1],ignore_samples=True,filter='all')
        #progress report
        sys.stdout.write('\r')        
        sys.stdout.write("Reading file {}\n".format(filename))
        sys.stdout.write("[%-20s] %d%%\n" % ('='*round((i+1)/total_file*20), round((i+1)/total_file*100)))
        sys.stdout.flush()
        
    #convert lists to data frame.
    events  = pd.concat(e, ignore_index=True)    
    messages= pd.concat(m, ignore_index=True)
    #remove saccade events.
    events = events.loc[events["type"] == "fixation"]    
    #get trialid messages to assign conditions to events
    events = events.merge(messages.loc[:,['trialid ', 'subject']],on='subject')
    
    #assign stimulus size to fixations
    dummy  = messages.loc[messages["DISPLAY_COORDS"].notnull(),["subject","DISPLAY_COORDS"]]
    events = events.merge(dummy,on='subject')
    
    return events
def sanity_checks(df):
    #check whether all fixations have same stimulus size.
    OK = False
    if len(np.unique(df.DISPLAY_COORDS.as_matrix())) == 1:
        OK = True
        return OK
    #check number of fixations per subjects, show results as a bar plot
    #check number of fixations per conditions, show results as 2d count matrix.

def fdm(df,bins=10,method='hist2d'):
    '''
        Computes a Fixation Density Map (FDM) based on fixations in the DataFrame.
        
        By default uses matplotlib's hist2d function. Specify which method to use 
        with METHOD argument. Possible methods are pandas, hist2d.
        
        BINS specifies the number of bins on the vertical axis, the horizonal bin
        size is deduced from stimulus size.
        
        Returns a 2D dataframe where each cell represents fixation counts at that 
        spatial location.
    '''
#    import matplotlib.pyplot as plt
#    import numpy as np
#    stim_size = stimulus_size(df)
#    if method == "hist2d":
#        return plt.hist2d(df["gavx"],df["gavy"],range=np.array(stim_size).reshape(2,2).T,bins=[12,16])
#        
def stimulus_size(df):
    '''
        Checks whether stimulus sizes are consistent across participants.
    '''
    if len(np.unique(df.DISPLAY_COORDS.as_matrix())) == 1:
        return df.DISPLAY_COORDS[0]
    else:
        print("DataFrame contains heterogenous set of stimulus sizes!")
        raise SystemExit

#def get_FDM()

#def plot_FDM()

#def pattern_similarity()

#def classification()
    #we will want to classify participants (not caring about conditions). 
    #or conditions 
