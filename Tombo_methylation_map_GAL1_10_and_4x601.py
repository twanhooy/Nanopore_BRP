import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO
import matplotlib as mpl
import scipy as sp
from scipy import linalg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
from scipy.stats import ks_2samp
import matplotlib.mlab as mlab

#tool to center the center point of the methylation bar to zero
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))

#function to read the reference genome in as a numpy array
def DNA_reading(DNA_fa):
    for record in SeqIO.parse(DNA_fa, "fasta"):
        DNA = (record.format("fasta").upper())#reads fasta file
    test_DNA = ([ list(word) for word in DNA ])
    DNA_arr_list = (test_DNA[34:-1])#cuts of header
    A=0
    C=0
    T=0
    G=0
    CpG = 0
    GpC = 0
    CpG_boolean = False
    GpC_boolean = False
    DNA_arr = np.empty(0)#empty array for DNA bases
    colors = np.empty(0)#empty array for color of eacht base
    loc_idex = -1 
    for q in DNA_arr_list:
        loc_idex +=1
        if q[0]=='A':#adds a A in blue to both arrrays
            A +=1
            DNA_arr = np.append(DNA_arr,q[0])
            colors = np.append(colors,'b')
            CpG_boolean = False
            GpC_boolean = False
        elif q[0]=='C':#adds a C in yellow to both arrrays
            C +=1
            colors = np.append(colors,'y')
            DNA_arr = np.append(DNA_arr,q[0])
            CpG_boolean = True#keep track of C in CpG
            if GpC_boolean==True: #count the amount of CpG's sites in the DNA
                GpC +=1
                GpC_boolean=False
            else:
                GpC_boolean=False
        elif q[0]=='G':#adds a G in green to both arrrays
            G +=1
            DNA_arr = np.append(DNA_arr,q[0])
            colors = np.append(colors,'g')
            GpC_boolean = True
            #print(loc_idex)
            if CpG_boolean==True: #count the amount of CpG's sites in the DNA
                CpG +=1
                CpG_boolean=False
            else:
                CpG_boolean=False
        elif q[0]=='T':#adds a T in red to both arrrays
            T +=1
            DNA_arr = np.append(DNA_arr,q[0])
            colors = np.append(colors,'r')
            CpG_boolean = False
            GpC_boolean = False
    return DNA_arr,colors,CpG,GpC
        
#extracts the positon, LLR and read_id from the text file with data
def reading(file):
    df = np.genfromtxt(file,skip_header=False)
    pos_float = df[:,0]
    stat = df[:,1]
    read_id_float = df[:,2]
    pos = pos_float.astype(int)
    read_id = read_id_float.astype(int)
    return pos,stat,read_id


#takes the mean of the methylation over 
def mean_freq(methyl_arr,DNA_arr,mean_number):
    length_mean = int(len(DNA_arr)/mean_number)
    mean_arr = np.zeros(length_mean)
    for i in range(length_mean):
        mean_arr[i] = (np.sum(methyl_arr[i*mean_number:(i+1)*mean_number]))/np.sum(methyl_arr[i*mean_number:(i+1)*mean_number]!=0)
    #Needed to acount for last entry error with mean over not a round number
    mean_arr[length_mean-1] = (np.sum(methyl_arr[mean_number*int(len(DNA_arr)/mean_number):len(DNA_arr)]))/np.sum(methyl_arr[mean_number*int(len(DNA_arr)/mean_number):len(DNA_arr)]!=0)
    return mean_arr

    
def perread_dataframe(read_id,pos,stat,DNA_arr,data,CpG,treshold,block_size):
    begin_read = read_id[0]
    methyl_arr = np.empty(len(DNA_arr))
    methyl_arr[:]=np.nan#gives an nan for every not C base in the array
    pos_arr = np.empty(len(DNA_arr))
    pos_arr[:] = np.nan
    length_arr = np.empty(0)
    read_num_arr = np.empty(0)
    for i in range(len(read_id)):#for every read one cycle
        #checks or the read id is the same as previous cycle
        boolean_element = (begin_read==read_id[i])
        if boolean_element==True:#if the read id is the same add to the array
            posit = pos[i]
            pos_arr[posit]=posit
            methyl_arr[posit]=stat[i]
        #if read id is different we filter out reads smaller than in:
        # In CpG context a fraction of total amount of CpG sites
       # In 5mC context can adjust to fraction of total amount of C sites 
        elif np.sum(np.abs(methyl_arr)>0)<int(treshold):
            methyl_arr = np.empty(len(DNA_arr))#reset for small reads
            methyl_arr[:]=np.nan
            pos_arr = np.empty(len(DNA_arr))
            pos_arr[:] = np.nan
            begin_read = read_id[i]
        elif data.shape[1]<block_size:
            #add read to dataframe if conditions are good
            data[begin_read]=methyl_arr
            length_arr = np.append(length_arr,(np.nanmax(pos_arr)-np.nanmin(pos_arr)))
            read_num_arr = np.append(read_num_arr,begin_read)
            methyl_arr = np.empty(len(DNA_arr))
            methyl_arr[:]=np.nan#reset
            pos_arr = np.empty(len(DNA_arr))
            pos_arr[:] = np.nan
            begin_read = read_id[i]
        else:
            break
    return data,read_num_arr

#gives base locations of CpG sites inside the reference genome
def c_finder(pos):
    cpg_arr = np.empty(0)
    for k in pos:
        checker = (cpg_arr==k)
        if np.sum(checker)==0:
            cpg_arr = np.append(cpg_arr,k)
    c_array_float = np.sort(cpg_arr)
    c_array = c_array_float.astype(int)
    return c_array

#plots histogram of the data
def plot_hist_function(df_mean):
    #histogram over the sum of methylation per read
    df_hist_ax0 = (df_mean.sum(axis=0))
    hist_ax0 = df_hist_ax0.hist(bins=100)
    plt.title('histogram over every read summation')
    plt.xlabel('sum over read i')
    plt.ylabel('frequency')
    plt.show()
    #mean over every CpG site
    #excluding zero because every not CpG site gives a zero in the histogram
    df_hist_ax1 = (df_mean.sum(axis=1))
    bin_arr =np.linspace(1,300,50)
    hist_ax1 = df_hist_ax1.hist(bins=bin_arr)
    plt.title('histogram over every base i')
    plt.xlabel('sum over base i')
    plt.ylabel('frequency')
    plt.show()

#makes plot off mean methylation over the whole genome
def normalized_plot_function(df_mean,DNA_arr):
    df_hist_ax1 = (df_mean.sum(axis=1))
    binned_arr = df_hist_ax1.values
    #normalization over total number of reads
    normalized_arr = binned_arr/df_mean.shape[1]
    mean_arr = mean_freq(normalized_arr,normalized_arr,10)
    fig, ax = plt.subplots(figsize=(24, 12))
    x_arr_methyl = np.linspace(0,len(DNA_arr),(len(normalized_arr)))
    plt.plot(x_arr_methyl,normalized_arr)
    #plt.axis(xmin=4700,xmax=4810,ymax=1,ymin=-1)
    plt.xlabel("base i")
    plt.ylabel("Methylated fraction")
    plt.title("GalLocus-in-plasmid+ SMAC-seq")
    plt.axis(xmin=0,xmax=len(DNA_arr))
    #plt.savefig("GalLocus_Plus_methyl_frac_plus.pdf")


#work in progress on PCA        
def Principle_Component_Analysis(data,comp_num,clus_num,loc_array,save_TRUE):
    #Make an numpy array from dataframe
    data_transposed = data.T
    X_full_mat = data.values
    n=data.shape[1]#specify the amount of reads
    p=len(loc_array)#amount of m5C/CpG sites in the perread statistics
    X = np.zeros([n,p])#empty array
    for i in range(n):
        for j in range(p):
            X[i,j] = X_full_mat[loc_array[j],i]
    X = np.nan_to_num(X)#needs zero for further analysis
    pca = PCA(n_components=comp_num)#computes the PCA
    pca.fit(X)# fit the PCA to the data
    Y = pca.transform(X)#gives an subset of the transformed data
    ex_variance=np.var(Y,axis=0)
    ex_variance_ratio = ex_variance/np.sum(ex_variance)
    #has to specify how much clusters you want to look for with clus_num
    kmeans = KMeans(n_clusters=clus_num)#Searches for clusters
    kmeans.fit(Y)#fit clusters to the reduced data
    center_points = (kmeans.cluster_centers_)#gives center of the cluster
    y_km = kmeans.fit_predict(Y)#gives different cluster in an array
    data_transposed['Cluster']= y_km
    #makes four subsets from the found cluster for later plotting
    subset_1 = (data_transposed.loc[data_transposed['Cluster'] == 0])
    subset_rm_1 = subset_1.drop(columns=['Cluster'])
    subset_1_finsihed = subset_rm_1.T
    subset_2 = (data_transposed.loc[data_transposed['Cluster'] == 1])
    subset_rm_2 = subset_2.drop(columns=['Cluster'])
    subset_2_finsihed = subset_rm_2.T
    subset_3 = (data_transposed.loc[data_transposed['Cluster'] == 2])
    subset_rm_3 = subset_3.drop(columns=['Cluster'])
    subset_3_finsihed = subset_rm_3.T
    subset_4 = (data_transposed.loc[data_transposed['Cluster'] == 3])
    subset_rm_4 = subset_4.drop(columns=['Cluster'])
    subset_4_finsihed = subset_rm_4.T
    fig = plt.figure(5,figsize=(10, 10))
    #plots four clusters, for less clusters the right amount of
    #plots needs to be turned on
    for i in range(3):
        if len(Y[y_km ==i,0])==subset_1_finsihed.shape[1]:
            plt.scatter(Y[y_km ==i,0], Y[y_km ==i,1], c='r',s=15,label='Full methylation')
    for i in range(3):
        if len(Y[y_km ==i,0])==subset_2_finsihed.shape[1]:
            plt.scatter(Y[y_km ==i,0], Y[y_km ==i,1], c='b',s=15,marker="^",label='No methylation')
    for i in range(3):
        if len(Y[y_km ==i,0])==subset_3_finsihed.shape[1]:
            plt.scatter(Y[y_km ==i,0], Y[y_km ==i,1], c='g',s=15,marker="s",label='GpC methylation')
    for i in range(4):
        if len(Y[y_km ==i,0])==subset_4_finsihed.shape[1]:
            plt.scatter(Y[y_km ==i,0], Y[y_km ==i,1], c='orange',s=15,marker="*",label='CpG methylation')
    plt.grid()
    tot_length = len(Y[y_km ==0,0])+len(Y[y_km ==1,0])+len(Y[y_km ==2,0])+len(Y[y_km ==3,0])
    #print the fractions of each cluster
    print('red',len(Y[y_km ==0,0])/tot_length)
    print('blue',len(Y[y_km ==1,0])/tot_length)
    print('green',len(Y[y_km ==2,0])/tot_length)
    print('orange',len(Y[y_km ==3,0])/tot_length)
    if save_TRUE==True:
        plt.savefig('GpC_CpG_PCA.png',dpi=500)
    plt.show()
    return Y,subset_1_finsihed,subset_2_finsihed,subset_3_finsihed,subset_4_finsihed
    
#make a plot for CpG against GpC
def CpG_GpC_selection(DNA_arr,data,save_TRUE):
    CpG_arr = np.empty(0)
    GpC_arr = np.empty(0)
    DNA_arr_min = np.ones(len(DNA_arr))
    DNA_arr_min = DNA_arr_min[:].astype(str)
    for base in range(len(DNA_arr)):
        if DNA_arr[base]=='G':
            DNA_arr_min[base]='C'
        elif DNA_arr[base]=='C':
            DNA_arr_min[base]='G'
        elif DNA_arr[base]=='A':
            DNA_arr_min[base]='T'
        elif DNA_arr[base]=='T':
            DNA_arr_min[base]='A'
    #searches CpG and GpC spots in the DNA array
    for j in range(len(DNA_arr)):
        if DNA_arr_min[j]=='C':
            if DNA_arr_min[j+1]=='G':
                loc_CpG = j
                CpG_arr = np.append(CpG_arr, loc_CpG)
        elif DNA_arr_min[j]=='G':
            if DNA_arr_min[j+1]=='C':
                loc_GpC = j+1
                GpC_arr = np.append(GpC_arr, loc_GpC)
    #makes a dataframe with only CpG methylations
    df_CpG = pd.DataFrame()
    for z in CpG_arr:
        df_CpG[int(z)]=(data.iloc[int(z)])
    df_CpG = df_CpG.T
    #makes a dataframe with only GpC methylations
    df_GpC = pd.DataFrame()
    for z in GpC_arr:
        df_GpC[int(z)]=(data.iloc[int(z)])
    df_GpC = df_GpC.T
    #takes the mean for every read over only the CpG and GpC dataframes
    CpG_sum_array = df_CpG.mean(axis=0)
    GpC_sum_array = df_GpC.mean(axis=0)
    #puts the dataframes into arrays
    CpG_PCA = CpG_sum_array.values
    GpC_PCA = GpC_sum_array.values
    #print(CpG_PCA.shape[0])#amount of reads
    Y = np.zeros([GpC_PCA.shape[0],2])#makes a 2 by reads array for clustering
    for k in range(CpG_PCA.shape[0]):
        Y[k,0]=CpG_PCA[k]
        Y[k,1]=GpC_PCA[k]
    kmeans = KMeans(n_clusters=3)#Searches for clusters
    Y = np.nan_to_num(Y)
    kmeans.fit(Y)#fit clusters to the reduced data
    y_km = kmeans.fit_predict(Y)
    #plotting both axis with the four clusters coloured
    mpl.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(4):
        if len(Y[y_km ==i,0])==np.sum(y_km ==0):
            plt.scatter(Y[y_km ==i,0], Y[y_km ==i,1], c='g',s=15,marker='P',label='Full methylation')
    for i in range(4):
        if len(Y[y_km ==i,0])==np.sum(y_km ==1):
            plt.scatter(Y[y_km ==i,0], Y[y_km ==i,1], c='r',s=15,marker='P',label='No methylation')
    for i in range(4):
        if len(Y[y_km ==i,0])==np.sum(y_km ==2):
            plt.scatter(Y[y_km ==i,0], Y[y_km ==i,1], c='b',s=15,marker='P',label='GpC methylation')
    for i in range(4):
        if len(Y[y_km ==i,0])==np.sum(y_km ==3):
            plt.scatter(Y[y_km ==i,0], Y[y_km ==i,1], c='r',s=15,marker='P',label='CpG methylation')
    plt.grid()
    plt.xlabel("Mean CpG methylation per read")
    plt.ylabel("Mean GpC methylation per read")
    if save_TRUE==True:
        plt.savefig('CpG_GALLOCUS.png',dpi=500)
    plt.show()
    #gives the fragments sizes
    data_CpGpC = pd.concat([df_CpG, df_GpC], axis=0, sort=False)
    tot_length = len(Y[y_km ==0,0])+len(Y[y_km ==1,0])+len(Y[y_km ==2,0])#+len(Y[y_km ==3,0])
    print(tot_length)
    print('red',len(Y[y_km ==0,0])/tot_length)
    print('blue',len(Y[y_km ==1,0])/tot_length)
    print('green',len(Y[y_km ==2,0])/tot_length)
    #print('orange',len(Y[y_km ==3,0])/tot_length)
    value_chi_square = [tot_length,len(Y[y_km ==0,0]),len(Y[y_km ==1,0]),len(Y[y_km ==2,0]),len(Y[y_km ==3,0])]
    return CpG_arr,GpC_arr,DNA_arr_min,data_CpGpC,value_chi_square


#makes a plot with the four subsets aranged
def plot_PCA_subsection(subset_1_finished,subset_2_finished,subset_3_finished,subset_4_finished,DNA_arr,save_TRUE):
    #reads the column names of the subset
    read_column_1 = (subset_1_finished.columns)
    read_column_1 = read_column_1.values
    read_column_2 = (subset_2_finished.columns)
    read_column_2 = read_column_2.values
    read_column_3 = (subset_3_finished.columns)
    read_column_3 = read_column_3.values
    read_column_4 = (subset_4_finished.columns)
    read_column_4 = read_column_4.values
    fig, (ax1) = plt.subplots(1,1, figsize=(14, 8))
    plt.tight_layout()
    cm = plt.cm.get_cmap('seismic_r')
    cm2 = plt.cm.get_cmap('viridis')
    mpl.rcParams.update({'font.size': 15})
    #tool to sort the subset in the right order for the right labeling
    select_column_array = np.append(read_column_1,read_column_2)
    select_column_array = np.append(select_column_array,read_column_3)
    select_column_array = np.append(select_column_array,read_column_4)
    select_subset_array = pd.concat([subset_1_finished,subset_2_finished,subset_3_finished,subset_4_finished],axis=1,sort=False)
    x_arr = np.linspace(0,len(DNA_arr),len(DNA_arr))
    norm = MidpointNormalize(vmin=-5, vmax=5, midpoint=0)
    norm2 = MidpointNormalize(vmin=-1, vmax=1, midpoint=0)
    g=0
    for k in range(select_subset_array.shape[1]):
        column_name = select_column_array[k]
        column_select =select_subset_array[column_name].values
        read_num = g*np.ones(len((DNA_arr)))
        g+=1
        p = ax1.scatter(x_arr,read_num,s=1,c=column_select,cmap=cm,norm=norm)
    plt.axis(xmin=0,xmax=len(DNA_arr),ymin=0,ymax=g)   
    plt.ylabel('Readnumber',fontsize=15)
    plt.xlabel('base index i',fontsize=15)
    fig.colorbar(p,label='Methylated                 Log Likelihood Ratio                 Unmethylated',ax=ax1,orientation='vertical')
    if save_TRUE==True:
        plt.savefig('Methylation_map.png',dpi=500)                  
    plt.show()
    return

def methylation_probability_density(Dataset_1,Dataset_2,DNA_arr,save_true,First_seq,Second_seq):
    CpG_arr = np.empty(0)
    GpC_arr = np.empty(0)
    unmety_C = np.empty(0)
    #reads the location of C in CpG  or GpC sites
    for j in range(len(DNA_arr)):
        if DNA_arr[j]=='C':
            if DNA_arr[j+1]=='G':
                loc_CpG = j
                CpG_arr = np.append(CpG_arr, loc_CpG)
            else:
                unmety_C = np.append(unmety_C,j)
        elif DNA_arr[j]=='G':
            if DNA_arr[j+1]=='C':
                loc_GpC = j+1
                GpC_arr = np.append(GpC_arr, loc_GpC)
    GpCpG_arr = np.concatenate([CpG_arr, GpC_arr[~np.isin(GpC_arr,CpG_arr)]])
    hist_arr = np.empty(0)
    #select the type of base you want to compare
    if First_seq=='GpC':
        First_seq = GpC_arr
    elif First_seq=='CpG':
        First_seq = CpG_arr
    elif First_seq=='GpCpG':
        First_seq = GpCpG_arr
    if Second_seq=='GpC':
        Second_seq = GpC_arr
    elif Second_seq=='CpG':
        Second_seq = CpG_arr
    elif Second_seq=='GpCpG':
        Second_seq = GpCpG_arr
    #makes a array with LLR of all C in reference
    for k in  First_seq:
        row = (Dataset_1.iloc[int(k)])
        row = row.values
        if np.nansum(row)==0:
            pass
        else:
            hist_arr = np.append(hist_arr,row)
    hist_arr_2 = np.empty(0)
    for h in  Second_seq:
        row2 = (Dataset_2.iloc[int(h)])
        row2 = row2.values
        if np.nansum(row2)==0:
            pass
        else:
            hist_arr_2 = np.append(hist_arr_2,row2)
    #remove nan values from the arrays
    hist_arr_2 = hist_arr_2[~np.isnan(hist_arr_2)]
    hist_arr = hist_arr[~np.isnan(hist_arr)]
    #makes the range of LLR values taken into account
    cutoff_arr = np.linspace(-10,10,201)
    fig, ax = plt.subplots(figsize=(10, 10))
    #plot to probability densities
    n_unmethy, bins_unmethy, patches_unmethy = plt.hist(hist_arr_2,bins=200,alpha = 0.75,density=True,color='blue')
    n_methy, bins_methy, patches_methy = plt.hist(hist_arr,bins=200,alpha = 0.75,density=True,color='red')
    plt.xticks(np.linspace(-10,10,9))
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(color='black', linestyle='dashed', linewidth=1)
    plt.xlabel("Log Likelihood Ratio")
    plt.ylabel("probability density")
    if save_true==True:
        plt.savefig('methyl_prob_density.png',dpi=500)
    plt.show()
    #make the cumulative distrubtion
    certainty_arr = np.empty(0)
    certainty_arr_not = np.empty(0)
    for g in range(len(cutoff_arr)):
        u = cutoff_arr[g]
        non_C_value = (np.sum(hist_arr_2<u)/len(hist_arr_2))
        certainty_arr_not = np.append(certainty_arr_not,non_C_value)
        C_value = (np.sum(hist_arr<u)/len(hist_arr))
        certainty_arr = np.append(certainty_arr,C_value)
    # makes the true positive rate
    procentuel_change_arr =certainty_arr/(certainty_arr_not+certainty_arr)
    #plotting the cumulative distrubtion
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax1.plot(cutoff_arr,certainty_arr,linewidth=5,color='red')
    ax1.plot(cutoff_arr,certainty_arr_not,linewidth=5,color='blue')
    plt.axis(xmin=-10,xmax=10,ymin=0,ymax=1)
    plt.yticks(np.linspace(0,1,9))
    ax1.plot(cutoff_arr,procentuel_change_arr,color='purple',linewidth=5)
    plt.gca().xaxis.grid(True)
    plt.grid(color='black', linestyle='dashed', linewidth=1)
    plt.xticks(np.linspace(-10,10,9))
    plt.xlabel("Log Likelihood Ratio")
    plt.ylabel("Cumulative fraction")
    plt.axvline(-2.5,color='black', linestyle='dashed', linewidth=1)
    plt.axvline(2.5,color='black', linestyle='dashed', linewidth=1)
    plt.axvline(-7.5,color='black', linestyle='dashed', linewidth=1)
    plt.axvline(7.5,color='black', linestyle='dashed', linewidth=1)
    plt.axvline(-5,color='black', linestyle='dashed', linewidth=1)
    plt.axvline(5,color='black', linestyle='dashed', linewidth=1)
    plt.axvline(0,color='black', linestyle='dashed', linewidth=1)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    if save_true==True:
        plt.savefig('culmative_llr_distrubtion.png',dpi=500)
    plt.show()
    return hist_arr,hist_arr_2


#Run the script to get all the data needed to make the plot        
def running(tombo_pr,dna_file,treshold,block_size):
    DNA_arr,colors,CpG,GpC = DNA_reading(dna_file)
    pos,stat,read_id = reading(tombo_pr)
    df = pd.DataFrame({'base': DNA_arr})
    mean_data,read_num_arr = perread_dataframe(read_id,pos,stat,DNA_arr,df,CpG,treshold,block_size)
    data_final = (mean_data.set_index('base'))
    loc_array = c_finder(pos)
    return data_final, read_num_arr,DNA_arr,loc_array,CpG,GpC

#list of different files used to plot the data
#The CP28 plasmid smaller data set
tombo_CP28_plus = 'tombo_CP28_per_read_plus_small_set.txt'
tombo_CP28_minus = 'tombo_CP28_per_read_minus_small_set.txt'
#The CP28 plasmid an extended data set
tombo_CP28_plus_extend = 'tombo_CP28_per_read_plus.txt'
tombo_CP28_minus_extend = 'tombo_CP28_per_read_minus.txt'  
#tombo detection on CpG on the GalLocus plasmid
tombo_Gal_plus = "tombo_GalLocus_per_read_plus.txt"
tombo_Gal_minus = "tombo_GalLocus_per_read_minus.txt"
#tombo detection on 5mC on the GalLocus plasmid
tombo_5mc_gal_plus = "tombo_GalLocus_per_read_plus_5mC.txt"
#DNA files
DNA_CP28_fa = "CP28.fa"
DNA_gal_fa = "GalLocus.fa"

#extracts data from the per read statistics
#specify the reference file to be used
DNA_reference = DNA_gal_fa
#specify the data file, which matches the reference file
TOMBO_file = tombo_5mc_gal_plus
#specify the length threshold to be used or the amount of sites to include
#around 2000 for 5mC around 200 for CpG mode
length_treshold = 2200
#specify the amount of reads taken into account
block_size = 200

data, read_num_arr,DNA_arr,loc_array,CpG,GpC = running(TOMBO_file,DNA_reference,length_treshold,block_size)

#plot a histogram with mean read methylation or the mean methylation per base
#plot_hist_function(df_mean)
#normalized_plot_function(df_mean,DNA_arr)

#PCA of the data to get four subsection
Y_return,subset_1_finished,subset_2_finished,subset_3_finished,subset_4_finished = Principle_Component_Analysis(data,2,4,loc_array,False)

#makes a plot of LLR on CpG sites against GpC sites
CpG_arr,GpC_arr,min_DNA_arr,data_CpGpC,value_chi_square = CpG_GpC_selection(DNA_arr,data,False)

#chi-square-test on the fraction sizes
tot_len, GpC_frac, non_frac,CpG_frac,full_frac = (value_chi_square)
p_full = 0.4444
p_GpC = 0.3333
p_CpG = 0.0733
p_no = 0.1488
chi_square_test = (full_frac-tot_len*p_full)**2/(tot_len*p_full)+(CpG_frac-tot_len*p_CpG)**2/(tot_len*p_CpG)+(GpC_frac-tot_len*p_GpC)**2/(tot_len*p_GpC)+(non_frac-tot_len*p_no)**2/(tot_len*p_no)
print(chi_square_test)

#gives the size of the different subsections
print(subset_1_finished.shape)
print(subset_2_finished.shape)
print(subset_3_finished.shape)
print(subset_4_finished.shape)

#plot_PCA_subsection(subset_1_finished,subset_3_finished,subset_4_finished,subset_2_finished,DNA_arr,CPG_GPC_colour)

#plot the probability densities for the subsets of the data
#select subset you want to compare
Dataset_1 = subset_1_finished
Dataset_2 = subset_2_finished
#save_true most been give True if you want to save the figure
save_true=False
#First_Seq and Second_seq specify the type of sites you want to compare
#there are 3 options GpC sites, CpG sites and for both use GpCpG sites
First_seq = 'GpCpG'
Second_seq = 'CpG'

methylation_probability_density(Dataset_1,Dataset_2,DNA_arr,save_true,First_seq,Second_seq)


     




















