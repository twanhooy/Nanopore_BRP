import h5py
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
from matplotlib import gridspec
from scipy import signal
import NucleosomePositionCore as NPC

#sort the fasta files in the right order
def DNA_name_arr(chrom):
    NAME_arr = [['0','0']]
    for i in range(1,17):
        name = ''
        if i <10:
            DNA = 'chr0'+str(i)+'.fsa'
        else:
            DNA = 'chr'+str(i)+'.fsa'
        for record in SeqIO.parse(DNA, "fasta"):
            DNA_rec = (record.format("fasta").upper())#reads fasta file
            test_DNA = ([ (word) for word in DNA_rec ])
            DNA_arr_list = (test_DNA[1:16])#cuts of header
        for g in range(len(DNA_arr_list)):
            name +=str(DNA_arr_list[g])
        name.upper()
        NAME_arr.append([str(i),name,DNA])
    GENOM_NAME = ((NAME_arr[chrom][1]))
    DNA_name = (NAME_arr[chrom][2])
    return GENOM_NAME,DNA_name

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

#reads the data from the hdf5 file
def data_reading(hdf5_shiporov,chromosome_number,strand_name,block_numb,x_min,x_max,lenght_bool,data_size):
    GENOM_NAME,DNA_name = DNA_name_arr(chromosome_number)
    reader_arr = np.empty(0)
    ls2b_arr= np.empty(0)
    names = 0
    #reading the chromosome reference file
    for record in SeqIO.parse(DNA_name, "fasta"):
        DNA = (record.format("fasta").upper())#reads fasta file
        test_DNA = ([ np.array(word) for word in DNA ])
        DNA_arr_list = (test_DNA[122:-1])#cuts of header
    df = pd.DataFrame({'base': DNA_arr_list})
    df_DNA = (df.where(df!='\n'))
    df_DNA = (df_DNA.dropna())
    pos = np.empty(0)
    stat = np.empty(0)
    read_id_names=np.empty(0)
    block_length=0
     #opens the HDF5 file
    with h5py.File(hdf5_shiporov,'r') as hdf:
        ls = list(hdf.keys())
        data = hdf.get('Statistic_Blocks')
        read_id = np.empty(0)
        for i in range(np.array(data).shape[0]):
            block = data.get('Block_'+str(i))
            block_set = np.array(block)
            ls2 = (block.attrs)
            ls2a = ls2.get('chrm')#chromosome name
            ls2a = ls2a.upper()
            ls2b = ls2.get('start')#block starting positon
            ls2c = ls2.get('strand')#gives the strand
            #checks or the right statistics block is selected
            if ls2a==GENOM_NAME and ls2c==strand_name and ls2b==block_numb:
                block_stat = np.array(block.get('block_stats'))
                read_ids_stat = np.array(block.get('read_ids'))
                block_length += (block_stat.shape[0])
                read_id = block_stat[0][2]
                ls2b_arr = np.append(ls2b_arr,ls2b)
                reader_arr = np.append(reader_arr,read_id)
                for j in range(block_stat.shape[0]):
                    read_id_num = block_stat[j][2]
                    #checks or it is still the same read
                    if read_id_num==read_id:
                        pos_numb = (block_stat[j][0])
                        #check or the read is located around the TSS
                        if x_min<= pos_numb <=x_max: 
                            pos = np.append(pos,pos_numb)
                            stat_numb = (block_stat[j][1])
                            stat = np.append(stat,stat_numb)
                        else:
                            pass
                    else:
                        methyl_arr = np.zeros(df_DNA.shape[0])
                        methyl_arr[:]=np.nan
                        #check the length of the read passes the threshold
                        if len(pos)<=lenght_bool:
                            #resetting the arrays
                            read_id = block_stat[j][2]
                            pos = np.empty(0)
                            stat = np.empty(0)
                            pos_numb = (block_stat[j][0])
                            pos = np.append(pos,pos_numb)
                            stat_numb = (block_stat[j][1])
                            stat = np.append(stat,stat_numb)
                        else:
                            for z in range(len(pos)):
                                pos = pos.astype(int)
                                methyl_arr[pos[z]]=stat[z]
                            df_DNA[str(names)+str(strand_name)]=methyl_arr
                            #checks or the amount of reads is not too big
                            if df_DNA.shape[1]<data_size:
                                names += 1
                                #resetting the arrays
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
    df_DNA = df_DNA.set_index('base')
    chrom_name = 'Chromosome '+str(chromosome_number)+' and strand '+str(strand_name)
    return df_DNA,read_id_names,chrom_name,DNA_arr_list

#sorts the data in plus and minus strand
def plotting_sorting(data,x_min,x_max,length_bool,plus_or_min,plot,DNA_arr_list,chrom_name):
    data_refined =pd.DataFrame({'base': DNA_arr_list})
    data_refined = (data_refined.where(data_refined!='\n'))
    data_refined = (data_refined.dropna())
    length_arr = np.empty(0)
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
        if length>=length_bool:
            read_label +=1
            data_refined[str(read_label)]=methyl_arr
    data_refined = data_refined.set_index('base')
    df_new = data_refined.iloc[x_min:x_max]
    print(length_arr)
    print(len(length_arr))
    print("----------")
    return df_new,DNA_arr_list

#calculates the NUS on a read
def nucleosome_exp(test_arr,treshold):
    nucl_arr = np.zeros(len(test_arr))
    for k in range(len(test_arr)-200):
        selected_arr = test_arr[k:k+147]
        selected_arr_bool = np.invert(np.isnan(selected_arr))
        Methylated_bases = (np.nansum(selected_arr<=-treshold))
        Un_Methylated_bases = (np.nansum(selected_arr>=-treshold))
        nucl_arr[k+73]=Un_Methylated_bases/(Methylated_bases+Un_Methylated_bases)
    return nucl_arr

#plots the methylation map and NUS of every read
def nucleosome_density(data,x_min,x_max,plus_cutoff,TSS,upper_bound,lower_bound,Gene_name,up_or_down,save_name,methy_threshold):
    cm1 = plt.cm.get_cmap('viridis')
    cm2 = plt.cm.get_cmap('seismic_r')
    cm3 = plt.cm.get_cmap('plasma_r')
    PCA_nucleo_df = pd.DataFrame()
    norm = MidpointNormalize(vmin=-5, vmax=5, midpoint=0)
    norm2 = MidpointNormalize(vmin=0.6, vmax=1, midpoint=0.8)
    norm3 = MidpointNormalize(vmin=lower_bound, vmax=upper_bound, midpoint=(lower_bound+(upper_bound-lower_bound)/2))
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(12, 12), gridspec_kw = {'height_ratios':[1,1,1]}, sharex=True)
    plt.tight_layout()
    x_arr = np.arange(x_min,x_max)
    read_arr_new = np.ones(len(x_arr))
    b_min = 0
    b_plus = 0
    #select the plus or minus strand reads
    for b in range(0,data.shape[1]):
        if b < plus_cutoff:
            test_arr = data[str(b_plus)+'+'].values
            b_plus +=1
        else:
            test_arr = data[str(b_min)+'-'].values
            b_min+=1
        #calculate the NUS on the read
        nucl_arr = nucleosome_exp(test_arr,methy_threshold)
        PCA_nucleo_df[b]=nucl_arr
        #plot the methylation and NUS of a single read
        p = ax2.scatter(x_arr,b*read_arr_new,s=50,c=nucl_arr,cmap=cm1,norm=norm2)
        d = ax1.scatter(x_arr,b*read_arr_new,c=test_arr,s=50,cmap=cm2,norm=norm)
    #mean NUS over the whole dataset    
    nucl_occup_mean = (PCA_nucleo_df.mean(axis=1))
    nucl_occup_mean = nucl_occup_mean.values
    nucl_std = PCA_nucleo_df.std(axis=1)
    nucl_std = nucl_std.values
    #plot the mean NUS and the confidence interval around it
    g = ax3.plot(x_arr,nucl_occup_mean,color='blue',linewidth=3)
    g1 = ax3.plot(x_arr,nucl_occup_mean+2*nucl_std/np.sqrt(PCA_nucleo_df.shape[1]),color='blue',linestyle='dashed',linewidth=0.5)
    g2 = ax3.plot(x_arr,nucl_occup_mean-2*nucl_std/np.sqrt(PCA_nucleo_df.shape[1]),color='blue',linestyle='dashed',linewidth=0.5)
    #print the TSS
    ax3.axvline(x=TSS,linestyle='dashed',color='g',linewidth=5)
    ax1.axvline(x=TSS,linestyle='dashed',color='g',linewidth=5)
    ax2.axvline(x=TSS,linestyle='dashed',color='g',linewidth=5)
    #give the direction of the gene
    if up_or_down=='up':
        ax3.annotate(Gene_name,ha = 'center', va = 'bottom',xytext = (TSS, upper_bound),xy = (TSS+100, upper_bound),arrowprops = {'facecolor' : 'green'},fontsize=16)
        ax1.annotate(Gene_name,ha = 'center', va = 'bottom',xytext = (TSS, data.shape[1]),xy = (TSS+100, data.shape[1]),arrowprops = {'facecolor' : 'green'},fontsize=16)
        ax2.annotate(Gene_name,ha = 'center', va = 'bottom',xytext = (TSS, data.shape[1]),xy = (TSS+100, data.shape[1]),arrowprops = {'facecolor' : 'green'},fontsize=16)
        ax1.axis(xmin=TSS-1000,xmax=TSS+1000,ymax=data.shape[1],ymin=0)
        ax2.axis(xmin=TSS-1000,xmax=TSS+1000,ymax=data.shape[1],ymin=0)
        ax3.axis(xmin=TSS-1000,xmax=TSS+1000,ymin=lower_bound,ymax=upper_bound)
        plt.xticks([TSS-1000,TSS-750,TSS-500,TSS-250,TSS,TSS+250,TSS+500,TSS+750,TSS+1000],['-1000','-750','-500','-250','0','250','500','750','1000'])
    elif up_or_down=='down':
        ax3.annotate(Gene_name,ha = 'center', va = 'bottom',xytext = (TSS, upper_bound),xy = (TSS-100, upper_bound),arrowprops = {'facecolor' : 'green'},fontsize=16)
        ax1.annotate(Gene_name,ha = 'center', va = 'bottom',xytext = (TSS, data.shape[1]),xy = (TSS-100, data.shape[1]),arrowprops = {'facecolor' : 'green'},fontsize=16)
        ax2.annotate(Gene_name,ha = 'center', va = 'bottom',xytext = (TSS, data.shape[1]),xy = (TSS-100, data.shape[1]),arrowprops = {'facecolor' : 'green'},fontsize=16)
        ax1.axis(xmin=TSS+1000,xmax=TSS-1000,ymax=data.shape[1],ymin=0)
        ax2.axis(xmin=TSS+1000,xmax=TSS-1000,ymax=data.shape[1],ymin=0)
        ax3.axis(xmin=TSS+1000,xmax=TSS-1000,ymin=lower_bound,ymax=upper_bound)
        plt.xticks([TSS+1000,TSS+750,TSS+500,TSS+250,TSS,TSS-250,TSS-500,TSS-750,TSS-1000],['-1000','-750','-500','-250','0','250','500','750','1000'])
    plt.colorbar(p,ax=ax1)
    plt.colorbar(d,ax=ax2)
    ax1.grid(color='black', linestyle='dashed', linewidth=1)
    ax2.grid(color='black', linestyle='dashed', linewidth=1)
    ax3.grid(color='black', linestyle='dashed', linewidth=1)
    plt.savefig(save_name+'.png',dpi=500)
    plt.show()
    return PCA_nucleo_df,nucl_occup_mean


def sorting_on_methyl(data,DNA_arr_list,x_min,x_max,plus_min):
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
    descent_arr = descent_arr.astype(int)
    for g in range(len(descent_arr)):
        test = data[str(descent_arr[g])+plus_min]
        methyl_sort_data[str(g)+plus_min] = test
    return methyl_sort_data

def running(hdf5_new,chrom,block_numb,plus_or_min,plot,x_min,x_max,length_bool,,data_size):
    data_first,read_id_names_plus,chrom_name,DNA_arr_list = data_reading(hdf5_new,chrom,plus_or_min,block_numb,x_min,x_max,length_bool,data_size)
    df_sorted,DNA_arr_list = plotting_sorting(data_first,x_min,x_max,length_bool,plus_or_min,plot,DNA_arr_list,chrom_name)
    return df_sorted,DNA_arr_list

def processing_6mA_data(HDF_file,chrom,block_loc,x_min,x_max,length_bool_plus,length_bool_min,data_size):
    df_sorted_plus,DNA_arr_list = running(HDF_file,chrom,block_loc,'+',False,x_min,x_max,length_bool_plus,data_size)
    df_sorted_min,DNA_arr_list = running(HDF_file,chrom,block_loc,'-',False,x_min,x_max,length_bool_min,data_size)
    return df_sorted_plus,df_sorted_min,DNA_arr_list

def Nucleosome_TOOL_calculator(DNA_arr_list,x_min,x_max):
    DNA_longer_endpoints =pd.DataFrame({'base': DNA_arr_list})
    DNA_longer_endpoints = (DNA_longer_endpoints.where(DNA_longer_endpoints!='\n'))
    DNA_longer_endpoints = (DNA_longer_endpoints.dropna())
    DNA_frag_test = (DNA_longer_endpoints[x_min-1000:x_max+1000])
    DNA_fragment = ''
    w = 147
    mu = -10.4
    B = 0.2
    period = 10.1
    for i in range(DNA_frag_test.shape[0]):
        DNA_fragment = DNA_fragment+str(DNA_frag_test.iloc[i][0])
    res = NPC.CalcNucPositions(DNA_fragment, w, mu, B, period)
    E_n = res[0]
    E = res[1]
    P = res[2]
    N = res[3]
    N = N[1000:-1000]
    return N