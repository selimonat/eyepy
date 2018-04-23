


#FDM is Fixation Density Maps.

#from pyedfread import edf


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
