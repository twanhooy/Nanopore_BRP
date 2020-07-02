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
import sys
from matplotlib import gridspec


hdf5_5mC = 'perread_Ship_Green_100plus_fullmethy.5mC.tombo.per_read_stats'

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
#GENOM_NAME,DNA_name = DNA_name_arr(7)
#print(GENOM_NAME,DNA_name)

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

#reading the HDF5 file and placing it in a dataframe
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
            if ls2a==GENOM_NAME and ls2b==block_numb and ls2c==strand_name:
                block_stat = np.array(block.get('block_stats'))
                read_ids_stat = np.array(block.get('read_ids'))
                block_length += (block_stat.shape[0])
                read_id = block_stat[0][2]
                reader_arr = np.append(reader_arr,read_id)
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
                        if len(pos)<=lenght_bool:
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
                            if df_DNA.shape[1]<data_size:
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
    df_DNA = df_DNA.set_index('base')
    return df_DNA,read_id_names,DNA_arr_list

#contracts the total reference genome to the sites around the TSS
def plotting_sorting(data,x_min,x_max,length_bool,plus_or_min,plot,DNA_arr_list):
    print(data)
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
    print(df_new)
    print(length_arr)
    print(len(length_arr))
    print("----------")
    return df_new,DNA_arr_list

#place the methylation sites on the correct CpG or GpC site
#inside the reference genome
def CpG_GpC_selection(data,plus_or_min,plot,DNA_arr_list):
    indexing = data.reset_index(level=['base'])
    index = indexing['base']
    DNA_arr = index.astype(str).values
    #refercing the bases for the minus strand
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
    df_GpC = pd.DataFrame()
    for z in GpC_arr:
        GpC_temp = (data.iloc[[int(z)]])
        GpC_temp=GpC_temp.values
        GpC_temp = (GpC_temp[0])
        df_GpC[str(int(z))]=GpC_temp
    df_GpC = df_GpC.T
    return CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list

#imbeds all the sites on the right location inside the plasmid
def CpG_GpC_sites_only_plot(CpG_arr,GpC_arr,df_CpG,df_GpC,x_min,x_max,DNA_arr_list,methy_threshold):
    #contract the CpG and GpC sites into one dataframe
    data = pd.concat([df_CpG, df_GpC], axis=0, sort=False)
    df_Gene = pd.DataFrame({'base': DNA_arr_list})
    df_Gene = (df_Gene.where(df_Gene!='\n'))
    df_Gene = (df_Gene.dropna())
    df_Gene = df_Gene.set_index('base')
    df_Gene=df_Gene.iloc[x_min:x_max]
    full_sites = np.append(CpG_arr,GpC_arr)
    full_sites  = full_sites.astype(int)
    #adds the CpG and GpC sites on the right C base in the reference genome
    for l in range(data.shape[1]):
        CpG_GpC_methyl = data[l]
        x_arr = np.arange(x_min,x_max)
        Full_site_arr = np.zeros(len(x_arr))
        Full_site_arr[:]=np.nan
        #constraints the sites to be added when a certain threshold is passed
        for k in range(len(full_sites)):
            if np.abs(CpG_GpC_methyl[k])>=methy_threshold:
                Full_site_arr[full_sites[k]]=CpG_GpC_methyl[k]
        df_Gene[l]=Full_site_arr
    return df_Gene

#sort the reads on the mean methylation per read
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

def running(hdf5_new,chrom,block_numb,plus_or_min,plot,x_min,x_max,length_bool,data_size):
    data_first,read_id_names_plus,DNA_arr_list = data_reading(hdf5_new,chrom,plus_or_min,block_numb,x_min,x_max,length_bool,data_size)
    df_sorted,DNA_arr_list = plotting_sorting(data_first,x_min,x_max,length_bool,plus_or_min,plot,DNA_arr_list)
    CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list = CpG_GpC_selection(df_sorted,plus_or_min,plot,DNA_arr_list)
    return CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list

def processing_data(hdf5,chrom,block_stat,x_min,x_max,length_bool_plus,length_bool_min,data_size,methy_threshold):
    CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list = running(hdf5,chrom,block_stat,'+',False,x_min,x_max,length_bool_plus,data_size)
    data_plus = CpG_GpC_sites_only_plot(CpG_arr,GpC_arr,df_CpG,df_GpC,x_min,x_max,DNA_arr_list,methy_threshold)
    CpG_arr,GpC_arr,df_CpG,df_GpC,DNA_arr_list = running(hdf5,chrom,block_stat,'-',False,x_min,x_max,length_bool_min,data_size)
    data_min = CpG_GpC_sites_only_plot(CpG_arr,GpC_arr,df_CpG,df_GpC,x_min,x_max,DNA_arr_list,methy_threshold)
    return data_plus,data_min,DNA_arr_list


