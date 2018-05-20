


#FDM is Fixation Density Maps.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
#from pyedfread import edf

def get_filelist(search_pattern):
    '''
    Find EDF files that will be used for the analysis and retrieves metadata about 
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
        #remove saccade events.
        e[i] = e[i].loc[e[i]["type"] == "fixation"]    
        #take only useful columns and include the meta data
        e[i] = e[i][list(file[1].keys())+['trial','gavx','gavy','start','end']]
        #progress report
        sys.stdout.write('\r')        
        sys.stdout.write("Reading file {}\n".format(filename))
        sys.stdout.write("[%-20s] %d%%\n" % ('='*round((i+1)/total_file*20), round((i+1)/total_file*100)))
        sys.stdout.flush()
        
    #convert lists to data frame.
    events  = pd.concat(e, ignore_index=True)    
    messages= pd.concat(m, ignore_index=True)
    #get trialid messages, and SYNCtime to assign conditions to events
    events = events.merge(messages.loc[:,['trialid ', 'SYNCTIME', 'py_trial_marker']],right_on='py_trial_marker',left_on='trial',how="left")
    #shift time stamps
    events.start = events.start - events.SYNCTIME
    events.end  = events.end  - events.SYNCTIME
    #remove prestimulus fixation points
    events = events[events.start > 0]
    #index fixations   
    def addfix(df):
        df["fix"] = range(df.shape[0])
        return df
    events = events.groupby("trial").apply(addfix)
    #drop useless columns
    events = events.drop(['SYNCTIME','py_trial_marker'],axis=1)
    
    #assign stimulus size to fixations
    dummy  = messages.loc[messages["DISPLAY_COORDS"].notnull(),["subject","DISPLAY_COORDS"]]
    events = events.merge(dummy,on='subject',how='left')
    
    #remove the 0th fixation as it is on the fixation cross.
    #add fixation weight.
    events.loc[:,'weight'] = 1;
    return events, messages
    #steps for cleaning.
    #crop fixations outside of a rect ==> should update stimulus size.

def sanity_checks(df):
    #check whether all fixation have the same stimulus size.
    #check number of fixations per subjects, show results as a bar plot.
    #check number of fixations per conditions, show results as 2d count matrix.
    #check for fixations outside the stimulus range.    
    1

def fdm(df,downsample=100):
    '''
        Computes a Fixation Density Map (FDM) based on fixations in the DataFrame.
        
        Uses matplotlib's hist2d function.
        
        DOWNSAMPLE is used to calculate the number of bins using
        bin = stim_size/downsample.
        
        Returns a 2D dataframe where each cell represents fixation counts at that 
        spatial location.
    '''    
    stim_size = stimulus_size(df)               #swap x and y so that hist2d gives nicely oriented FDMs.
    fdm_range = stim_size.reshape(2,2).T        #size of the count matrix
    fdm_bins = (stim_size[[2,3]]+1 )/downsample #number of bins    
    fdm, xedges, yedges, bla = plt.hist2d(df["gavx"],df["gavy"],range=fdm_range,bins=fdm_bins)
    return pd.DataFrame(fdm)
    #return fdm, xedges, yedges
    
def plot(df,path_stim='',downsample=100):
    '''
    Plots fixation counts and optinally overlays on the stimulus"
    '''
    #solve the problem with spyder that prevents two images to be overlaid.
    img=plt.imread(path_stim)
    plt.imshow(img,alpha=.5)    
    
    count,x,y = fdm(df,downsample)
    plt.imshow(count.T,extent=[x[0],x[-1],y[0],y[-1]],alpha=.5)    
    plt.show()    
#        
def stimulus_size(df):
    '''
        Checks whether stimulus sizes are consistent across participants. Throws 
        an error if not.
        (Note: This is slightly too slow I have the impression)
    '''
    sizes = np.unique(df.DISPLAY_COORDS.as_matrix())
    if len(sizes) == 1:
        return np.array(sizes[0])
    else:
        print("DataFrame contains heterogenous set of stimulus sizes!")
        raise SystemExit


def pattern_similarity(df):
    FPSA = 1-df.groupby("subject").apply(fdm).unstack(level=1).T.corr()
    plt.imshow(FPSA.values)
    plt.show()
    return FPSA

#def classification()
    #we will want to classify participants (not caring about conditions). 
    #or conditions 
