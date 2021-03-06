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
import math
from pyedfread import edf        
import sys

from sklearn import (manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection)
from scipy.spatial.distance import pdist, squareform
from time import time

from joblib import Memory

memory = Memory(cachedir='/tmp/', verbose=1)


def get_filelist(search_pattern):
    '''    
    Returns a list of tuples that stores path and metadeta about the EDF files found 
    in the SEARCH_PATTERN (the path where your project resides).
    
    For each EDF file, the output tuple contains its path and metadata extracted 
    from the path string. Metadata here means participant and run ids.
    
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

#@memory.cache()
def get_df(filelist,filter_fun=None,new_rect=None):
    '''
    Reads all EDF files in the filelist (see get_filelist) tuple with pyedfread 
    and returns a merged dataframe where each row is one fixation.
    
    The output dataframe consists of Event and Message dataframes (outputs of 
    pyedfread module, see pyedfread). 
    
    Metadata is used to label each row with participant and run id.
    (see get_filelist).
    
    If no FILTER_FUN is provided, Message DataFrame is directly appended to 
    Event DataFrame. In some cases one might desire to preprocess Message 
    DataFrame before appending. In that case, the result of FILTER_FUN(messages) 
    is appended.
    
    Specify NEWRECT to discard pixels outside of a square central region. NEWRECT 
    of 500 would discard all data outside of a square of 500x500 pixels.
        
    See get_filelist to know more about filelist format.
    See pyedfread for more information on the output dataframes.
    '''
    
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
        
        #take only useful columns, include the meta data columns as well.
        E             = E[list(file[1].keys())+['trial','gavx','gavy','start','end']]        
        
        #get SYNCTIME (i.e. stim onsets) 
        E             = E.merge(M.loc[:,['SYNCTIME', 'py_trial_marker']],
                                right_on='py_trial_marker',
                                left_on='trial',
                                how="left")
        
        #correct time stamps with SYNCTIME
        E.start       = E.start - E.SYNCTIME
        E.end         = E.end   - E.SYNCTIME
        #remove prestimulus fixations
        E             = E[E.start > 0]
        #index fixations
        E             = E.groupby("trial").apply(addfix)
        #get display coordinates and calibration values        
        E             = E.append(M.loc[M.py_trial_marker == -1,['validation_result:','subject','run']])

        #add the display coordinates.        
        rect = M.loc[M.py_trial_marker == -1,['DISPLAY_COORDS']].values        
        E['DISPLAY_COORDS']  = np.repeat(rect,E.shape[0],axis=0)        
        
        #drop useless columns
        E             = E.drop(['SYNCTIME','py_trial_marker'],axis=1)
        #get what we want from messages using filter_fun or simply discard messages.
        if filter_fun != None:
            m = filter_fun(M)
            e[i]  = E.merge(m,left_on='trial',right_on='py_trial_marker',how='left')
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
    events.loc[:,'weight'] = 1
    
    #Check whether all data has same DISPLAY_COORDINATES 
    #Overwrite DISPLAY_COORDINATES if required. 
    events = set_rect(events,new_rect)
    rect   = get_rect(events)
    
#    #Remove out of range fixation data    
    valid_fix     = (events.gavx >= rect[0]) & \
                    (events.gavx <= rect[2]) & \
                    (events.gavy >= rect[1]) & \
                    (events.gavy <= rect[3]) | events.trial.isnull()
                        
    events        = events.loc[valid_fix,:]
#                    
    
    return events 
    #remove the 0th fixation as it is on the fixation cross.
    #crop fixations outside of a rect ==> should update stimulus size.
    #rename fields

def sanity_checks(df):
    """
    Conducts sanity checks on the data and reports. (1) calibration error. (2)
    number of trials. DF must come from get_df function.
    """
    
    #check for calibration quality, detect badly calibrated participants
    V    = df[['subject','validation_result:']].dropna()
    V['validation_result:'] = V['validation_result:'].map(lambda x: x[0])        
    error = V['validation_result:'].values
    bad   = V.loc[V["validation_result:"] > .5,"subject"]
    
    #plt.style.use('fivethirtyeight')
    
    fig, (ax_calib,ax_count,ax_hist) = plt.subplots(1,3,figsize=(24, 6), dpi= 80, facecolor='w', edgecolor='k')
                
    
    ax_calib.hist(error,50)
    ax_calib.set_xlabel('Average Error ($^\circ$)')
    ax_calib.set_ylabel('# Participants')
    ax_calib.set_title('Calibration Quality')    
    ax_calib.axvline(.5, ls='--', color='r')
    ax_calib.spines["right"].set_visible(0)
    ax_calib.spines["top"].set_visible(0)
    ax_calib.title.set_fontsize(20)
    ax_calib.xaxis.label.set_fontsize(20)
    
    ax_calib.yaxis.label.set_fontsize(20)
    ax_calib.tick_params(labelsize=15)
    ax_calib.locator_params(axis='y', nbins=4)
    ax_calib.grid(color='k', linestyle='-', linewidth=.5,alpha=.1)
    
    report = "There are {} participants ({}) with a calibration error\nthat is more than \
    half a degree. You might consider\nexcluding them from the analysis".format(len(bad),bad.values)
    print(report)
        
    #check number of fixations per conditions, show results as 2d count matrix.
    fix_count = df.pivot_table(index=['subject'],columns='condition',values='trial',aggfunc=lambda x: len(np.unique(x)))
    
    ax_count.imshow(fix_count.values)    
    ax_count.set_xlabel('Conditions')    
    ax_count.set_ylabel('Participants')    
    ax_count.set_title('#Fixations')    
    ax_count.title.set_fontsize(20)
    ax_count.xaxis.label.set_fontsize(20)
    ax_count.yaxis.label.set_fontsize(20)
    ax_count.tick_params(labelsize=15)

    
    data = df.groupby(['subject']).apply( lambda x: (x['subject'].unique()[0],x.shape[0]) )
    s,c  = zip(*data)
    
    ax_hist.bar(s,c)
    ax_hist.set_xlabel('Participant ID')
    ax_hist.xaxis.label.set_fontsize(20)    
    ax_hist.set_ylabel('# Fixations')
    ax_hist.yaxis.label.set_fontsize(20)
    ax_hist.set_title('Number of Fixations')        
    ax_hist.title.set_fontsize(20)
    ax_hist.spines["right"].set_visible(0)
    ax_hist.spines["top"].set_visible(0)        
    ax_hist.tick_params(labelsize=15)
    ax_hist.locator_params(axis='y', nbins=4)
    ax_hist.grid(color='k', linestyle='-', linewidth=.5,alpha=.1)
    
    
    
    #check for fixations outside the stimulus range.
    #add scatter matrix with x,y values per subject
    colors = df['subject'].transform(lambda x: (x-min(x))/(max(x)-min(x))).transform(lambda x: [x,0,1-x])
    
        
    pd.tools.plotting.scatter_matrix(df[['gavx','gavy','fix']],c=colors, diagonal='hist',alpha=0.3,marker='o')    
	
    #check whether all fixation have the same stimulus size
    check_rect(df)
    
def get_rect_size(df):
    """
    Returns the span of the rect in df as a 1 x 2 tuple where 
    (vertical span, horizontal span)
    """
    rect = get_rect(df)
    return np.array((rect[3]-rect[1]+1,rect[2]-rect[0]+1))
       
def check_rect(df):
    """
        returns True if rect size is consistent
    """
    if len(np.unique(df.DISPLAY_COORDS)) == 1:
        print("Consistency check: All files have same stimulus size (rect).")        
        return True
    else:
        return False

def get_rect(df):
    """
        returns the rect if consistent.
    """    
    if check_rect(df):
        rect = df["DISPLAY_COORDS"].iat[0]
        return rect    
        
def set_rect(df,new_size=None):
    '''
        Checks whether all stimulus rectangles are consistent across participants.
        Else, throws RuntimeError.
        If new_value given, replaces the old value. New_value represents the size of 
        square rect, centrally positioned on the previous rect.
    '''    
    rect = get_rect(df)             # all rects present in the DF
    if new_size is not None:        # replace rect with a new one.        
        print('Will overwrite rect with new value: {}'.format(rect[0]))        
        rect                 = new_rect(rect,new_size)
        df["DISPLAY_COORDS"] = rect.tolist() * df.shape[0]        
    return df

def new_rect(old_rect,w):
    '''
    Returns a new centrally located rect (as np row) with size w based on the previous one.
    
    '''    
    if w % 2 == 0:
        midpoint_x = (old_rect[2])/2
        midpoint_y = (old_rect[3])/2
        return np.array([[math.ceil(midpoint_x-w/2),
                math.ceil(midpoint_y-w/2),
                math.floor(midpoint_x+w/2),
                math.floor(midpoint_y+w/2)]])
    else:
        print("w must be an even number")    


def fdm(df,downsample=100):
    '''
        Computes a Fixation Density Map (FDM) based on fixations in the DataFrame.
        
        Uses matplotlib's hist2d function.
        
        DOWNSAMPLE is used to calculate the number of bins using
        bin = stim_size/downsample.
        
        Returns a 2D DataFrame where each cell represents fixation counts at that 
        spatial location.
        
        Example:
           fdm =  eyepy.fdm(df)
    '''    
    rect      = get_rect(df)                     #size of rect
    rect      = np.array(rect)
    fdm_range = rect.reshape(2,2).T              #size of the count matrix
    fdm_range[:,1] = fdm_range[:,1] + 1
    stim_size = get_rect_size(df)                #stim size
    fdm_bins  = stim_size/downsample #number of bins                

    fdm,xedges,yedges = np.histogram2d(df["gavx"].dropna().values,
                                       df["gavy"].dropna().values,
                                       range=fdm_range,
                                       bins=fdm_bins)
     
    #return fdm, xedges, yedges
    #return pd.DataFrame(fdm)                     # another option is just to return numpy array
    df = pd.DataFrame(fdm, index=xedges[:-1], columns=yedges[:-1])
    df.index.name   = "rows"
    df.columns.name = "columns"
    return df

    
def plot_group(G):
    """
    Produce a grid of subplot with and plot single participant average
    FDMs.
    G = df.groupby(['subject']);
    eyepy.plot_subject(G)
    """    
    
    tpanel = np.ceil(np.sqrt(10))
    for panel,g in enumerate(G):
        group_name = g[0]
        df         = g[1]
        plt.subplot(tpanel,tpanel,panel+1)
        plt.title(group_name)
        plot(df)
    
def plot(df,path_stim='',downsample=100):
    '''
        Plots fixation counts and optionally overlays on the stimulus"
    '''
    #solve the problem with spyder that prevents two images to be overlaid.
    if path_stim != '':
        img = plt.imread(path_stim)
        plt.imshow(img,alpha=.5)    
    
    count = fdm(df,downsample)
    x     = count.index.values[0], count.index.values[-1]
    y     = count.columns.values[0], count.columns.values[-1]
    plt.imshow(count.T,extent=[x[0],x[-1],y[0],y[-1]],alpha=.5)    
    plt.show()    

def pattern_similarity(G):
    """
        Computes a similarity matrix based on the group G. To compute a similarity 
        matrix across participant G should be df.groupby('subjects').
    """          
    FPSA = pdist(G.apply(fdm).unstack(level=1),metric='correlation')    
    return FPSA

def dendrogram(D):
    
    import scipy.cluster.hierarchy as sch
    
    D = squareform(D)
    
    fig,(axdendro,axmatrix) = plt.subplots(2,1)    
    axdendro.set_position([0.3,0.71,0.6,0.2])
    #axdendro                = fig.add_axes([0.09,0.1,0.2,0.8])
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    axdendro.spines["top"].set_visible(0)
    axdendro.spines["right"].set_visible(0)        

    Y = sch.linkage(D, method='centroid')
    Z = sch.dendrogram(Y, orientation='top',ax=axdendro)

    index = Z['leaves']
    D = D[index,:]
    D = D[:,index]
    
    axmatrix.set_position([0.3,0.1,0.6,0.6])    
    im = axmatrix.imshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    
    # Display and save figure.
    fig.show()    


    
# Scale and visualize the embedding vectors
def plot_embedding(X, target, title=None):
    """
    Plots the results of different manifold learning algorithms. Assumes 2-d.
    """
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(target[i]),
                 color=plt.cm.Set1(target[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})    
    
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def group2dataset(G):
    """
        Transforms a Group (i.e. dataframe.groupby object) to a standard dataset 
        format, such as MNIST dataset that can be downloaded.
        out["data"] is one row per sample and a column per feature (pixel or pc)
        out["target"] contains the group labels values. If DataFrame is grouped
        by subjects, it contaisns subject indices.
    """
    data   = G.apply(fdm).unstack().values
    labels = list(G.groups.keys());
    
    out = {'COL_NAMES' : [],
           'DESCR'  :  '',
           'data'   : data,
           'targets': labels}
    
    return out

def decompose(dataset):
    
    X  = dataset["data"]
    y  = dataset["targets"]
    t0 = time()
    
    X_decomposed = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    
    #X_decomposed = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_decomposed = tsne.fit_transform(X)
    
    
    embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack")
    X_decomposed = embedder.fit_transform(X)

    
    clf          = manifold.MDS(n_components=2, n_init=1, max_iter=100)
    X_decomposed = clf.fit_transform(X)
    
    
    plot_embedding(X_decomposed,y,
               "Principal Components projection of the digits (time %.2fs)" %
               (time() - t0))
    return X_decomposed
    
def kmeans(G):
    
    from sklearn.cluster import KMeans
    #transform FDM to (n_samples,n_features), where n_samples is the number of
    # subjects, trials etc, and n_features is the number of pixels.
    x = G.apply(fdm).unstack().values

    t_cluster = 3
    
    # K Means Cluster
    model = KMeans(n_clusters=t_cluster)
    model.fit(x)

    #largest number of items in any cluster
    _,t = np.unique(model.labels_,return_counts=True)
    max_count = max(t)
    
    #    
    f, axarr = plt.subplots(t_cluster, max_count)
    citem    = np.zeros([t_cluster,1],dtype=np.int8)
    for i,g in enumerate(G):
            this_cluster         = model.labels_[i]
            citem[this_cluster] += 1            
            print(this_cluster)
            print(citem[this_cluster])
            plt.sca(axarr[this_cluster,int(citem[this_cluster]-1)])
            plot(g[1])


        
def get_data(df):
    '''
    returns FDM data as [sample,feature] matrix together with labels.
    '''
    G         = df.groupby(['subject','trial'])
    t_sample  = len(G)
    t_feature = len(fdm(G.first()).unstack())
    data      = np.zeros([t_sample,t_feature])
    label     = np.zeros([t_sample,1])
    c = 0;
    for name,group in G:        
        label[c] = name[0]
        data[c,] = fdm(group).unstack()
        c=c+1;
    return data,label


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
