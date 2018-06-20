"""
Field name conventions:
    subject
    run
    trial
    condition
    x
    y
    start
    stop
    fix
    weight
"""

#FDM is Fixation Density Maps.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
#from pyedfread import edf

def get_filelist(search_pattern):
    '''    
    Returns a list of tuples that stores path and metadeta about EDF files found 
    in the SEARCH_PATTERN.
    
    For each EDF file found tuple contains its path, participant and run number.
    
    Metadata is recovered using the nested get_subject and get_run functions. 
    You might need to change those depending on how you stored your data. In 
    case this fails (which will most likely happen), create your own method to 
    generate a tuple for your specifics.
    
    Returns: 
        a list of tuples in the form [('filename', {'subject': N,'run':M})]
    
    Example:
    search_pattern = "/mnt/data/project_FPSA_FearGen/data/**/p02/**/data.edf"
    eyepy.get_filelist(search_pattern)
    [('/mnt/data/project_FPSA_FearGen/data/sub027/p02/eye/data.edf',
      {'run': 2, 'subject': 27}),...    
    '''
    import glob
    import re
    #two nested functions used to recover subject and run identities.
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

def get_fixmat(filelist,filter_fun=None):
    '''
    Reads all EDF files with pyedfread and returns a merged dataframe.
    
    Final dataframe consists of Event and Message dataframes (outputs of pyedfread).
    
    Metadata is used to label DataFrames from individual subjects and runs (see get_filelist).
    
    If no FILTER_FUN is provided, Message DataFrame is directly appended as is to events. 
    In some cases one might desire to preprocess Message DataFrame before appending. 
    In that case, the result of FILTER_FUN(messages) is appended.
        
    See get_filelist to know more about filelist format.
    See pyedfread for more information on the output dataframes.
    '''
    from pyedfread import edf        
    import sys
    total_file = len(filelist)
    print("Receivied {} EDF files".format(total_file))
    #init 3 variables that will accumulate data frames as a list
    e = [None] * total_file
    m = [None] * total_file
    #will be used to index fixation ranks below in df.groupby.apply combo
    def addfix(df):
        df["fix"] = range(df.shape[0])
        return df
    #Call pyedfread and concat different data frames as lists.
    for i,file in enumerate(filelist):
        filename                  = file[0]        
        _, E, M       = edf.pread(filename,meta=file[1],ignore_samples=True,filter='all')
        #just to be sure not to have any spaces on column names (this could be moved to pyedfread)
        E.rename(columns=str.strip,inplace=True)
        M.rename(columns=str.strip,inplace=True)
        #remove saccade events.
        E             = E.loc[E["type"] == "fixation"]
        #remove out of range fixation
        rect          = M.DISPLAY_COORDS.dropna()[0]
        valid_fix     = (E.gavx >= rect[0]) & (E.gavx <= rect[2]) & (E.gavy >= rect[1]) & (E.gavy <= rect[3])
        E             = E.loc[valid_fix,:]
        #take only useful columns, include the meta data columns as well.
        E             = E[list(file[1].keys())+['trial','gavx','gavy','start','end']]
        #get SYNCTIME (i.e. stim onsets) 
        E             = E.merge(M.loc[:,['SYNCTIME', 'py_trial_marker']], right_on='py_trial_marker',left_on='trial',how="left")
            #correct time stamps with SYNCTIME
        E.start       = E.start - E.SYNCTIME
        E.end         = E.end   - E.SYNCTIME
            #remove prestimulus fixations
        E             = E[E.start > 0]
        #index fixations
        E             = E.groupby("trial").apply(addfix)
        #get display coordinates and calibration values
        E             = E.append(M.loc[M.py_trial_marker == -1,['DISPLAY_COORDS', 'validation_result:','subject','run']])                
        #drop useless columns
        E             = E.drop(['SYNCTIME','py_trial_marker'],axis=1)
        #get what we want from messages
        if filter_fun != None:
            m = filter_fun(M)
            e[i]          = E.merge(m,left_on='trial',right_on='py_trial_marker',how='left')
        else:
            e[i]  = E
        
        #progress report
        sys.stdout.write('\r')        
        sys.stdout.write("Reading file {}\n".format(filename))
        sys.stdout.write("[%-20s] %d%%\n" % ('='*round((i+1)/total_file*20), round((i+1)/total_file*100)))
        sys.stdout.flush()
        
    #convert lists to data frame.
    events  = pd.concat(e, ignore_index=True)                    
    #add fixation weight (todo: add it only to trials > 0)
    events.loc[:,'weight'] = 1;    
    return events 
    #remove the 0th fixation as it is on the fixation cross.
    #crop fixations outside of a rect ==> should update stimulus size.
    #rename fields

def sanity_checks(df):
    """
    conducts sanity checks on the data and reports. (1) calibration error. (2)
    number of trials.
    """
    
    #check for calibration quality
    V= df[['subject','validation_result:']].dropna()
    V['validation_result:'] = V['validation_result:'].map(lambda x: x[0])        
    error = V['validation_result:'].values
    bad   = V.loc[V["validation_result:"] > .5,"subject"]
    
    plt.figure(figsize=(12, 8), dpi= 80, facecolor='w', edgecolor='k')
    plt.subplot(2,2,1)
    plt.hist(error)
    plt.xlabel('Average Error ($^\circ$)')
    plt.title('Calibration Quality')
    ymin, ymax = plt.ylim()
    plt.plot(np.repeat(np.mean(error),2),[ymin, ymax])
    plt.box('off')
    
    report = "There are {} participants ({}) with a calibration error\nthat is more than \
    half a degree. You might consider\nexcluding them from the analysis".format(len(bad),bad.values)
    print(report)
    #check number of fixations per conditions, show results as 2d count matrix.
    fix_count = df.pivot_table(index=['subject'],columns='condition',values='trial',aggfunc=lambda x: len(np.unique(x)))
    plt.subplot(2,2,2)
    plt.imshow(fix_count)
    plt.xlabel('Conditions')    
    plt.ylabel('Participants')
    plt.colorbar()
    plt.title('#Fixations per Condition')
    #check for fixations outside the stimulus range.
    #add scatter matrix with x,y values per subject
    colors = df['subject'].transform(lambda x: (x-min(x))/(max(x)-min(x))).transform(lambda x: [x,0,1-x])
    pd.tools.plotting.scatter_matrix(df[['gavx','gavy','fix']],c=colors,figsize=(15, 15), diagonal='hist',alpha=0.3,marker='o')
	#check whether all fixation have the same stimulus size
    stimulus_size    
    

def fdm(df,downsample=100,stim_size=[0,0,1599,1199]):
    '''
        Computes a Fixation Density Map (FDM) based on fixations in the DataFrame.
        
        Uses matplotlib's hist2d function.
        
        DOWNSAMPLE is used to calculate the number of bins using
        bin = stim_size/downsample.
        
        Returns a 2D dataframe where each cell represents fixation counts at that 
        spatial location.
        
        Example:
            eyepy.fdm(df)            
    '''    
    stim_size = np.array(stim_size)
    fdm_range = stim_size.reshape(2,2).T        #size of the count matrix
    fdm_bins  = (stim_size[[2,3]]+1 )/downsample #number of bins    
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
    sizes = np.unique(df.DISPLAY_COORDS.dropna())
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

def PCA(df,groupby,explained_variance=.95,downsample=100):
    '''
    Computes eigenvalue decomposition of eye-movement patterns based on fixation 
    density maps stored in the df.groupby(groupby).
    Returns components that are sufficient to explain explained_variance% of 
    variance (default = 95%).
    Plots components. 
    Use this function to visually investigate PCA components.
    '''
    from sklearn import decomposition
    M   = df.groupby(groupby).apply(fdm,downsample=downsample).unstack().T
    pca = decomposition.PCA(explained_variance)
    pca.fit(M)
    pca.transform(M)
    return pca

from sklearn.base import BaseEstimator, TransformerMixin
class FDM_generator(BaseEstimator, TransformerMixin):
    '''
    Generates fixation density maps with a resolution as hyper-parameter.
    '''    
    def __init__(self,resolution=40):
        self.resolution = resolution
    def fit(self,X,y=None):        
        return self
    def transform(self,X,y=None):        
        density_maps   = X.groupby(y).apply(eyepy.fdm,downsample=self.resolution).unstack().T
        ##convert it to values
        return density_maps
    
#from sklearn.pipeline import Pipeline
#from sklearn.pipeline import Pipeline
#num_pipeline = Pipeline([('fdm', FDM_generator()),
#                         ('pca', PCA(explained_variance)),
#                         ('std_scaler', StandardScaler()),
#                         ])            
    
    
#def classification()
    #we will want to classify participants (not caring about conditions). 
    #or conditions 
