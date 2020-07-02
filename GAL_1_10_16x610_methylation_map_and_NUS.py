import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO
import matplotlib as mpl
import scipy as sp
import scipy.stats as ss
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import sys
from matplotlib import gridspec
import NucleosomePositionCore as NPC
from scipy.stats import t  
from mpl_toolkits.axes_grid1 import make_axes_locatable

#function to normalize the colourbars used
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

#reads the reference file and the methylation data from the hdf5 file
def data_reading(hdf5_file,chromosome_file,chromosome_name,strand_name,block_numb,x_min,x_max,lenght_bool,MIN_CUT,MAX_CUT,block_size):
    DNA_name = chromosome_file
    names = 0
    for record in SeqIO.parse(DNA_name, "fasta"):
        DNA = (record.format("fasta").upper())#reads fasta file
        test_DNA = ([ np.array(word) for word in DNA ])
        #cuts of header but can vary with FASTA file
        DNA_arr_list = (test_DNA[49:-1])
    df = pd.DataFrame({'base': DNA_arr_list})
    #remove spaces from the fasta file
    df_DNA = (df.where(df!='\n'))
    df_DNA = (df_DNA.dropna())
    #setting up arrays for the data reading
    pos = np.empty(0)
    max_pos = np.empty(0)
    min_pos = np.empty(0)
    len_pos_frag = np.empty(0)
    stat = np.empty(0)
    read_id_names=np.empty(0)
    read_id = np.empty(0)
    block_length=0
    #reading the hdf5 file
    with h5py.File(hdf5_file,'r') as hdf:
        ls = list(hdf.keys())
        #get the first layer from the HDF5 file
        data = hdf.get('Statistic_Blocks')
        for i in range(np.array(data).shape[0]):
            #open a specific block
            block = data.get('Block_'+str(i))
            block_attributes = (block.attrs)
            #selects plasmid name or chromosome id of block
            block_chrom = block_attributes.get('chrm')
            block_chrom = block_chrom.upper()
            #selects start positon of block
            block_start = block_attributes.get('start')
            #selects strand of the block
            block_strand = block_attributes.get('strand')
            print(block_chrom,block_start,block_strand)
            # selects specific block with specified id and strand
            if block_chrom==chromosome_name and  \
            block_strand==strand_name and block_start==block_numb:
                #extrack statistics and read_id number
                block_stat = np.array(block.get('block_stats'))
                read_ids_stat = np.array(block.get('read_ids'))
                read_id = block_stat[0][2]
                for j in range(block_stat.shape[0]):
                    read_id_num = block_stat[j][2]
                    if read_id_num==read_id:
                        pos_numb = (block_stat[j][0])
                        if x_min<= pos_numb <=x_max: 
                            pos = np.append(pos,pos_numb)
                            stat_numb = (block_stat[j][1])
                            stat = np.append(stat,stat_numb)
                        else:
                            pass
                    else:
                        methyl_arr = np.zeros(df_DNA.shape[0])
                        methyl_arr[:]=np.nan
                        if len(pos)<lenght_bool:
                            read_id = block_stat[j][2]
                            pos = np.empty(0)
                            stat = np.empty(0)
                            pos_numb = (block_stat[j][0])
                            pos = np.append(pos,pos_numb)
                            stat_numb = (block_stat[j][1])
                            stat = np.append(stat,stat_numb)
# specify the location of the cutting enzyme used on the plasmid
                        elif (MIN_CUT-20)<np.min(pos)<(MIN_CUT+20) and (MAX_CUT-20)<np.max(pos)<(MAX_CUT+20):
                            for z in range(len(pos)):
                                pos = pos.astype(int)
                                methyl_arr[pos[z]]=stat[z]
                            df_DNA[str(names)+str(strand_name)]=methyl_arr
                            max_pos = np.append(max_pos,np.max(pos))
                            min_pos = np.append(min_pos,np.min(pos))
                            len_pos_frag = np.append(len_pos_frag,(np.max(pos)-np.min(pos)))
                            #amount of reads used in further analysis
                            if df_DNA.shape[1]<block_size:
                                names += 1
                                read_id_names = np.append(read_id_names,read_ids_stat[read_id])
                                read_id = block_stat[j][2]
                                pos = np.empty(0)
                                stat = np.empty(0)
                                pos_numb = (block_stat[j][0])
                                pos = np.append(pos,pos_numb)
                                stat_numb = (block_stat[j][1])
                                stat = np.append(stat,stat_numb)
                            else:
                                break
                        else:
                            read_id = block_stat[j][2]
                            pos = np.empty(0)
                            stat = np.empty(0)
                            pos_numb = (block_stat[j][0])
                            pos = np.append(pos,pos_numb)
                            stat_numb = (block_stat[j][1])
                            stat = np.append(stat,stat_numb)
    df_DNA = df_DNA.set_index('base')
    return df_DNA,read_id_names,chrom_name,DNA_arr_list,max_pos,min_pos,len_pos_frag


def plotting_sorting(data,x_min,x_max,length_bool,plus_or_min,plot,DNA_arr_list,chrom_name):
    data_refined =pd.DataFrame({'base': DNA_arr_list})
    data_refined = (data_refined.where(data_refined!='\n'))
    data_refined = (data_refined.dropna())
    length_arr = np.empty(0)
    if plot==True:
        fig, ax = plt.subplots(figsize=(30, 3))
        cm = plt.cm.get_cmap('RdGy')
    read_label = 0
    for g in range(data.shape[1]):
        if plus_or_min=='+':
            methyl = data[str(g)+'+']
        elif plus_or_min=='-':
            methyl = data[str(g)+'-']
        methyl_arr = methyl.values
        subset_methyl = methyl_arr[x_min:x_max]
        bool_arr = (np.isnan(subset_methyl)==False)*np.ones(len(subset_methyl))
        winner = np.argwhere(bool_arr==1)
        if len(winner)>0:
            max_win = winner[len(winner)-1]
            min_win = winner[0]
            length = int(np.sum(bool_arr))
            length_arr = np.append(length_arr,length)
        else:
            length=0
        if length>length_bool:
            read_label +=1
            data_refined[str(read_label)]=methyl_arr
            if plot==True:
                x_arr = np.arange(0,data.shape[0])
                norm = norm = MidpointNormalize(vmin=-10, vmax=10, midpoint=0)
                read_num = read_label*np.ones(data.shape[0])
                p = ax.scatter(x_arr,read_num,c=methyl_arr,s=50,cmap=cm,norm=norm)
    data_refined = data_refined.set_index('base')
    df_new = data_refined.iloc[x_min:x_max]
    return df_new,DNA_arr_list,length_arr

def CpG_GpC_selection(data,plus_or_min,plot,DNA_arr_list):
    indexing = data.reset_index(level=['base'])
    index = indexing['base']
    DNA_arr = index.astype(str).values
    #reverses the reference for the minus strand
    if plus_or_min=='-':
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
        DNA_arr = DNA_arr_min
    CpG_arr = np.empty(0)
    GpC_arr = np.empty(0)
    #searches CpG and GpC spots in the DNA array
    for j in range(len(DNA_arr)-1):
        if DNA_arr[j]=='C':
            if DNA_arr[j+1]=='G':
                loc_CpG = j
                CpG_arr = np.append(CpG_arr, loc_CpG)
        elif DNA_arr[j]=='G':
            if DNA_arr[j+1]=='C':
                loc_GpC = j+1
                GpC_arr = np.append(GpC_arr, loc_GpC)
    #makes a dataframe with only CpG methylations
    df_CpG = pd.DataFrame()
    for z in CpG_arr:
        CpG_temp = (data.iloc[[int(z)]])
        CpG_temp=CpG_temp.values
        CpG_temp = (CpG_temp[0])
        df_CpG[str(int(z))]=CpG_temp
    df_CpG = df_CpG.T
    #makes a dataframe with only GpC methylations
    df_GpC = pd.DataFrame(
    for z in GpC_arr:
        GpC_temp = (data.iloc[[int(z)]])
        GpC_temp=GpC_temp.values
        GpC_temp = (GpC_temp[0])
        df_GpC[str(int(z))]=GpC_temp
    df_GpC = df_GpC.T
    return CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list

#combines the CpG and GpC sites in one dataframe
def CpG_GpC_sites_only_plot(CpG_arr,GpC_arr,df_CpG,df_GpC,x_min,x_max,DNA_arr_list):
    data = pd.concat([df_CpG, df_GpC], axis=0, sort=False)
    df_Gene = pd.DataFrame({'base': DNA_arr_list})
    df_Gene = (df_Gene.where(df_Gene!='\n'))
    df_Gene = (df_Gene.dropna())
    df_Gene = df_Gene.set_index('base')
    print(df_Gene.shape)
    df_Gene=df_Gene.iloc[x_min:x_max]
    print(df_Gene.shape)
    full_sites = np.append(CpG_arr,GpC_arr)
    full_sites  = full_sites.astype(int)
    for l in range(data.shape[1]):
        CpG_GpC_methyl = data[l]
        x_arr = np.arange(x_min,x_max)
        Full_site_arr = np.zeros(len(x_arr))
        Full_site_arr[:]=np.nan
        for k in range(len(full_sites)):
            if np.abs(CpG_GpC_methyl[k])>=0:
                Full_site_arr[full_sites[k]]=CpG_GpC_methyl[k]
        df_Gene[l]=Full_site_arr
    return df_Gene

#Calculate the Nucleosome-Unmethylated-Score on a single read
def nucleosome_exp(test_arr,treshold):
    nucl_arr = np.zeros(len(test_arr))
    for k in range(len(test_arr)-150):
        selected_arr = test_arr[k:k+147]
        selected_arr_bool = np.invert(np.isnan(selected_arr))
        Methylated_bases_arr = ((selected_arr<=-treshold))
        Un_Methylated_bases_arr = ((selected_arr>=-treshold))
        Un_methy_sum = np.nansum(Un_Methylated_bases_arr)
        Methy_sum = np.nansum(Methylated_bases_arr)
        nucl_arr[k+73]=Un_methy_sum/(Methy_sum+Un_methy_sum)
    return nucl_arr

#plots a methylation map and the NUS score per read
def nucleosome_density(data,x_min,x_max,plus_cutoff,thresold,subset_name,column_name_arr):
    cm1 = plt.cm.get_cmap('viridis')
    cm2 = plt.cm.get_cmap('seismic_r')
    PCA_nucleo_df = pd.DataFrame()
    norm  = MidpointNormalize(vmin=-5, vmax=5, midpoint=0)
    norm2  = MidpointNormalize(vmin=0.4, vmax=1, midpoint=0.7)
    fig, (ax1, ax2,ax3) = plt.subplots(3,1, figsize=(12, 12), gridspec_kw = {'height_ratios':[1,1,1]}, sharex=True)
    plt.tight_layout()
    x_arr = np.arange(x_min,x_max)
    read_arr_new = np.ones(len(x_arr))    
    b_plus = 0
    b_minus = 0
    b_subset = 0
    #can be used to plot subsets of the data from PCA analysis
    if subset_name==True:
        for v in column_name_arr:
            test_arr = data[v].values
            nucl_arr = nucleosome_exp(test_arr,thresold)
            PCA_nucleo_df[b_subset]=nucl_arr
            test_arr = test_arr[x_min:x_max]
            nucl_arr = nucl_arr[x_min:x_max]
            p = ax2.scatter(x_arr,b_subset*read_arr_new,s=50,c=nucl_arr,cmap=cm1,norm=norm3)
            d = ax1.scatter(x_arr,b_subset*read_arr_new,c=test_arr,s=20,cmap=cm2,norm=norm)
            b_subset +=1
    else:
        for b in range(0,data.shape[1]):
            if b<plus_cutoff:
                test_arr = data[str(b_plus)+str('+')].values
                b_plus +=1
            elif b>=plus_cutoff:
                test_arr = data[str(b_minus)+str('-')].values
                b_minus +=1
            #calculating the NUS on single reads with 2.5 the set threshold
            nucl_arr = nucleosome_exp(test_arr,thresold)
            PCA_nucleo_df[b]=nucl_arr
            test_arr = test_arr[x_min:x_max]
            nucl_arr = nucl_arr[x_min:x_max]
            #nucleosome position plotter
            p = ax2.scatter(x_arr,b*read_arr_new,s=50,c=nucl_arr,cmap=cm1,norm=norm2)
            # plotting the indivudual methylation sites
            d = ax1.scatter(x_arr,b*read_arr_new,c=test_arr,s=20,cmap=cm2,norm=norm)
    #the mean NUS over all the reads        
    nucl_occup_mean = (PCA_nucleo_df.mean(axis=1))
    nucl_occup_mean = nucl_occup_mean.values
    nucl_occup_mean = nucl_occup_mean[x_min:x_max]
    #the std on the NUS of every read
    nucl_occup_std = (PCA_nucleo_df.std(axis=1))
    nucl_occup_std = nucl_occup_std.values
    nucl_occup_std = nucl_occup_std[x_min:x_max]
    # plotting mean off nucleosome positioner with confidence interval as
    # mu +- z_95(~2)*sigma/sqrt(n) with n the amount of reads as d.o.f.
    g = ax3.plot(x_arr,nucl_occup_mean,color='blue')
    g = ax3.plot(x_arr,nucl_occup_mean+nucl_occup_std*2/np.sqrt(PCA_nucleo_df.shape[1]),color='blue',linestyle='dashed',linewidth=0.5)
    g = ax3.plot(x_arr,nucl_occup_mean-nucl_occup_std*2/np.sqrt(PCA_nucleo_df.shape[1]),color='blue',linestyle='dashed',linewidth=0.5)
    ax1.axis(xmin=x_min,xmax=x_max,ymax=data.shape[1],ymin=0)
    ax2.axis(xmin=x_min,xmax=x_max,ymax=data.shape[1],ymin=0)
    #change the ymin here to see the NUS better
    ax3.axis(xmin=x_min,xmax=x_max,ymin=0.5,ymax=1)
    #plt.xlabel("base index i")
    ax1.grid(color='black', linestyle='dashed', linewidth=1)
    ax2.grid(color='black', linestyle='dashed', linewidth=1)
    ax3.grid(color='black', linestyle='dashed', linewidth=1)
    plt.savefig("methy_map_NUS.png",dpi=500)
    plt.show()
    return PCA_nucleo_df,nucl_occup_mean,nucl_occup_std


def sorting_on_methyl(data,DNA_arr_list,plus_minus,x_min,x_max):
    methyl_sort_data = pd.DataFrame()
    methyl_sort_data = pd.DataFrame({'base': DNA_arr_list})
    methyl_sort_data = (methyl_sort_data.where(methyl_sort_data!='\n'))
    methyl_sort_data = (methyl_sort_data.dropna())
    methyl_sort_data = methyl_sort_data.set_index('base')
    methyl_sort_data = methyl_sort_data.iloc[x_min:x_max]
    mean_data = data.mean(axis=0)
    mean_data = mean_data.values
    sorted_mean = np.sort(mean_data)#sort all the reads
    descent_arr = np.zeros(len(mean_data))
    for k in range(len(descent_arr)):
        bool_arr = (sorted_mean[k]==mean_data)*np.ones(len(mean_data))  
        idex = np.argwhere(bool_arr==1)
        descent_arr[k] = idex   
    for g in range(len(descent_arr)):
        test = data[descent_arr[g]]
        methyl_sort_data[str(g)+str(plus_minus)] = test
    return methyl_sort_data

#running all the codes for one strand of the data
def running(hdf5_new,chrom_file,chrom_name,block_numb,plus_or_min,plot,x_min,x_max,length_bool,MIN_CUT,MAX_CUT):
    data_first,read_id_names_plus,chrom_name,DNA_arr_list,max_pos,min_pos,len_pos_frag = data_reading(hdf5_new,chrom_file,chrom_name,plus_or_min,block_numb,x_min,x_max,length_bool,MIN_CUT,MAX_CUT)
    df_sorted,DNA_arr_list,length_arr = plotting_sorting(data_first,x_min,x_max,length_bool,plus_or_min,plot,DNA_arr_list,chrom_name)
    CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list = CpG_GpC_selection(df_sorted,plus_or_min,plot,DNA_arr_list)
    return df_sorted,CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list,length_arr,max_pos,min_pos,len_pos_frag

#processe the data from both strands
def processing_data(hdf5,chrom_file,chrom_name,block_stat,x_min,x_max,length_bool_plus,length_bool_min,CUT_MIN,CUT_MAX):
    df_sorted_plus,CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list,length_arr_plus,max_pos_max,min_pos_max,len_pos_frag_max = running(hdf5,chrom_file,chrom_name,block_stat,'+',False,x_min,x_max,length_bool_plus,CUT_MIN,CUT_MAX)
    data_plus = CpG_GpC_sites_only_plot(CpG_arr,GpC_arr,df_CpG,df_GpC,x_min,x_max,DNA_arr_list)
    df_sorted_min,CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list,length_arr_min,max_pos_min,min_pos_min,len_pos_frag_min = running(hdf5,chrom_file,chrom_name,block_stat,'-',False,x_min,x_max,length_bool_min,CUT_MIN,CUT_MAX)
    data_min = CpG_GpC_sites_only_plot(CpG_arr,GpC_arr,df_CpG,df_GpC,x_min,x_max,DNA_arr_list)
    return data_plus,data_min,DNA_arr_list

#converter used in Nucleosome_TOOL_calculator
def index2base(base):
    q=''
    if base==0:
        q = 'A'
    elif base==1:
        q = 'C'
    elif base==2:
        q = 'G'
    elif base==3:
        q = 'T'
    return q

#tool to calculate sequence based nucleosome probabilities
def Nucleosome_TOOL_calculator(DNA_arr_list,x_min,x_max):
    DNA_longer_endpoints =pd.DataFrame({'base': DNA_arr_list})
    DNA_longer_endpoints = (DNA_longer_endpoints.where(DNA_longer_endpoints!='\n'))
    DNA_longer_endpoints = (DNA_longer_endpoints.dropna())
    DNA_frag_test = (DNA_longer_endpoints[x_min:x_max])
    DNA_fragment = ''
    #nucleosome size
    w = 147
    #chemical potential used
    mu = -10.4
    #amplitude used 
    B = 0.2
    #period of the nucleosome
    period = 10.1
    #used to add random DNA to the begin and end of window to plot nucleosomes
    # on the edge of the window
    for k in range(DNA_frag_test.shape[0]):
        DNA_fragment = DNA_fragment+str(DNA_frag_test.iloc[k][0])
    DNA_longer = np.random.rand(1000)
    DNA_longer = DNA_longer*4
    DNA_longer = (DNA_longer.astype(int))
    telomer_end = ''
    for z in range(len(DNA_longer)):
        base = index2base(DNA_longer[z])
        telomer_end = telomer_end+base
    DNA_plus_end = telomer_end+DNA_fragment+telomer_end
    res = NPC.CalcNucPositions(DNA_plus_end, w, mu, B, period)
    E_n = res[0]
    E = res[1]
    P = res[2]
    N = res[3]
    N = N[1000:-1000]
    return N

#plots the mean NUS for the different treated reconstitued fractions
def plot_reconstitued_GAL_RAF_fractions(nucl_occup_mean_RECON,nucl_occup_std_RECON,PCA_nucleo_df_RECON,nucl_occup_mean_GAL,nucl_occup_std_GAL,PCA_nucleo_df_GAL,nucl_occup_mean_RAF,nucl_occup_std_RAF,PCA_nucleo_df_RAF,SAVE_TRUE):
    x_arr = np.arange(2249,7586)
    fig, ax1,ax2,ax3 = plt.subplots(3,1, figsize=(10, 5), gridspec_kw = {'height_ratios':[1,1,1]}, sharex=True, sharey=True)
    plt.tight_layout()
    ax1.plot(x_arr,nucl_occup_mean_RECON,color='blue',linestyle='solid',linewidth=2)
    ax1.fill_between(x_arr,nucl_occup_mean_RECON+1.96*nucl_occup_std_RECON/np.sqrt(PCA_nucleo_df_RECON.shape[1]),nucl_occup_mean_RECON-1.96*nucl_occup_std_RECON/np.sqrt(PCA_nucleo_df_RECON.shape[1]),color='blue',alpha=0.3)
    ax1.axis(ymax=1,ymin=0.75,xmin=2249,xmax=7586)
    ax2.plot(x_arr,nucl_occup_mean_GAL,color='red',linestyle='solid',linewidth=2)
    ax2.fill_between(x_arr,nucl_occup_mean_GAL+1.96*nucl_occup_std_GAL/np.sqrt(PCA_nucleo_df_GAL.shape[1]),nucl_occup_mean_GAL-1.96*nucl_occup_std_GAL/np.sqrt(PCA_nucleo_df_GAL.shape[1]),color='red',alpha=0.3)
    ax2.axis(ymax=1,ymin=0.75,xmin=2249,xmax=7586)
    ax3.plot(x_arr,nucl_occup_mean_RAF,color='green',linestyle='solid',linewidth=2)
    ax3.fill_between(x_arr,nucl_occup_mean_RAF+1.96*nucl_occup_std_RAF/np.sqrt(PCA_nucleo_df_RAF.shape[1]),nucl_occup_mean_RAF-1.96*nucl_occup_std_RAF/np.sqrt(PCA_nucleo_df_RAF.shape[1]),color='green',alpha=0.3)
    ax3.axis(ymax=1,ymin=0.75,xmin=2249,xmax=7586)
    ax1.axvline(6000,linestyle='dashed',linewidth=2,color=(1,51/255,0))
    ax2.axvline(6000,linestyle='dashed',linewidth=2,color=(1,51/255,0))
    ax3.axvline(6000,linestyle='dashed',linewidth=2,color=(1,51/255,0))
    ax1.axvline(4348,linestyle='dashed',linewidth=2,color=(179/255,85/255,163/255))
    ax2.axvline(4348,linestyle='dashed',linewidth=2,color=(179/255,85/255,163/255))
    ax3.axvline(4348,linestyle='dashed',linewidth=2,color=(179/255,85/255,163/255))
    plt.xticks(np.arange(2500,8000,500))
    plt.yticks(np.arange(0.75,1,0.05))
    ax1.grid(color='black', linestyle='dashed', linewidth=1)
    ax2.grid(color='black', linestyle='dashed', linewidth=1)
    ax3.grid(color='black', linestyle='dashed', linewidth=1)
    if SAVE_TRUE=True:
        plt.savefig('RECON_GAL_RAF.svg')
    plt.show()


#the HDF5 files used in our own analysis
hdf5 = 'prs_GAL_610_redone.5mC.tombo.per_read_stats'
hdf5_new = 'perread_plus_min_compare.5mC.tombo.per_read_stats'
hdf5_test = 'pr_test_realign.5mC.tombo.per_read_stats'

#here the fasta name and the name from the HDF5 file should be selected of
#the plasmid your intrested in
chrom_file = 'GalLocus-in-plasmid.fa'
chrom_name = 'GALLOCUS-IN-PLASMID(WITHLOOPS)'
x_max = 9768
x_min = 0
block_start = 0 #starting position of the statistics block default is 0
plus_strand_threshold =1000
minus_strand_threshold = 1000
#specify the cutsite location on the reference genome
Cutsite_lower = 69
Cutside_higher = 7883
#the amount of reads to include in the dataframe
df_size = 2000

hdf5,chrom_file,chrom_name,block_stat,x_min,x_max,length_bool_plus,length_bool_min,CUT_MIN,CUT_MAX
df_sorted_plus,df_sorted_min,data_RECON,RECON_min,DNA_arr_list = processing_data(hdf5,chrom_file,chrom_name,block_start,x_min,x_max,plus_strand_threshold,minus_strand_threshold,Cutsite_lower,Cutside_higher,df_size)
#sorts the reads on mean methylation
methyl_sort_data_RECON = sorting_on_methyl(data_RECON,DNA_arr_list,'+',x_min,x_max)
#specify the cutoff between plus and minus strand
plus_cutoff_RECON=methyl_sort_data_RECON.shape[1]
#makes the methylation map and the NUS on single reads
# the last to entries are only needed when a PCA subsection is being plottedd
#setting the treshold for the NUS to count a base as methylated/unmethylated 
methy_threshold =2.5
PCA_nucleo_df_RECON,nucl_occup_mean_RECON,nucl_occup_std_RECON = nucleosome_density(methyl_sort_data_RECON,2249,7586,plus_cutoff_RECON,methy_threshold,False,np.empty(0)) 



