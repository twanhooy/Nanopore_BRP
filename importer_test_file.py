import NucleosomePositionCore as NPC
import matplotlib.pyplot as plt
import GreenLeaf_data_tombo as GL_5mC
import GreenLeaf_tombo_6mA_methy_ as GL_6mA
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


#combines the data from the 5mC sites and the 6mA sites
def combiner_5mC_6mA(data_plus,data_min,df_sorted_plus,df_sorted_min):
    reset_base = df_sorted_plus.reset_index()
    base_arr = reset_base['base']
    plus_data_arr = np.nansum(np.dstack((data_plus.values,df_sorted_plus.values)),2)
    columns_plus = []
    for g in range(data_plus.shape[1]):
        name_str = str(g)+'+'
        columns_plus.append(name_str)
    df_plus = pd.DataFrame(plus_data_arr,columns=columns_plus)
    df_plus = df_plus.replace(0.0, np.nan)
    df_plus['base']=base_arr
    df_plus = df_plus.set_index('base')
    minus_data_arr = np.nansum(np.dstack((data_min.values,df_sorted_min.values)),2)
    columns_min = []
    for h in range(data_min.shape[1]):
        name_str = str(h)+'-'
        columns_min.append(name_str)
    df_minus = pd.DataFrame(minus_data_arr,columns=columns_min)
    df_minus = df_minus.replace(0.0, np.nan)
    df_minus['base']=base_arr
    df_minus = df_minus.set_index('base')
    methyl_plus = GL_6mA.sorting_on_methyl(df_plus,DNA_arr_list,x_min,x_max,'+')
    methyl_min = GL_6mA.sorting_on_methyl(df_minus,DNA_arr_list,x_min,x_max,'-')
    #methyl_tot_data = pd.concat([methyl_plus,methyl_min],axis=1)
    plus_cutoff=methyl_plus.shape[1]
    return methyl_plus,methyl_min,plus_cutoff

def PLOT_NUS_GAL1(PCA_nucleo_df_plus,single_mol_numb,save_name):
    fig, ax1 = plt.subplots(figsize=(20, 6)) 
    plt.rcParams.update({'font.size': 20})
    #select single molecule read and plot these
    single_molecule = PCA_nucleo_df_plus[single_mol_numb]
    ax1.plot(x_arr,single_molecule,color='b',linewidth=2)

    #potting mean NUS of plus and minus strand
    #ax1.plot(x_arr,nucl_occup_mean_min,color='b',linewidth=3)
    ax1.plot(x_arr,nucl_occup_mean_plus,color='r',linewidth=3)
    
    #calculate standard deviation on the mean
    nucl_occup_std_plus = PCA_nucleo_df_plus.std(axis=1)
    #nucl_occup_std_min = PCA_nucleo_df_min.std(axis=1)
    
    #calculate the 95 confidence interval around the mean
    # mu +- Z_95(.96)*std/sqrt(n)
    ax1.fill_between(x_arr,nucl_occup_mean_plus-(1.96*nucl_occup_std_plus)/np.sqrt(PCA_nucleo_df_plus.shape[1]),nucl_occup_mean_plus+(1.96*nucl_occup_std_plus)/np.sqrt(PCA_nucleo_df_plus.shape[1]),facecolor='r',alpha=0.5)
    #ax1.fill_between(x_arr,nucl_occup_mean_min-(1.96*nucl_occup_std_min)/np.sqrt(PCA_nucleo_df_min.shape[1]),nucl_occup_mean_min+(1.96*nucl_occup_std_min)/np.sqrt(PCA_nucleo_df_min.shape[1]),facecolor='b',alpha=0.5)
    #give position of gene features like TATA box etc.
    plt.axvline(5860-6000,color=(104/255,104/255,255/255),linestyle='solid',linewidth=7)
    plt.axvline(5654-6000,color=(104/255,181/255,104/255),linestyle='solid',linewidth=7)
    plt.axvline(5553-6000,color=(104/255,181/255,104/255),linestyle='solid',linewidth=7)
    plt.axvline(5575-6000,color=(104/255,181/255,104/255),linestyle='solid',linewidth=7)
    plt.axvline(5590-6000,color=(104/255,181/255,104/255),linestyle='solid',linewidth=7)
    plt.axvline(6000-6000,color=(31/255,119/255,186/255),linestyle='dashed',linewidth=5)
    plt.axvline(5332-6000,color='purple',linestyle='dashed',linewidth=5)
    #annotate the gene name in the 
    #plt.annotate('GAL1',ha = 'center', va = 'bottom',xytext = (TSS, 1),xy = (TSS-50,1),arrowprops = {'facecolor' : 'green'},fontsize=16)
    #plt.xticks([TSS-1000,TSS-750,TSS-500,TSS-250,TSS,TSS+250,TSS+500,TSS+750,TSS+1000],['-1000','-750','-500','-250','0','250','500','750','1000'])
    plt.grid(color='black', linestyle='dashed', linewidth=1,axis='both')
    plt.axis(xmin=-1000,xmax=1000,ymax=1,ymin=0.75)
    #save figure
    plt.savefig(save_name+'.svg')
    plt.show() 
    return

#Specify the Transcription start site of the specific gene
TSS = 463434
x_min = TSS-1000-150
x_max = TSS+1000+150
#selects the block were the TSS is located
block_stat=450000
#select chromosome number
chrom=4
#the amount of selected reads
data_size=100
#The threshold for adding a base in the methylation map
#this is a different threshold than in the nucleosome calling
methy_threshold=0 
#select length of the fragments with the numbers being the amount of 
#C in the fragment for 5mC and A for 6mA, this can differ per strand
length_bool_5mC_plus = 462
length_bool_5mC_min = 453
length_bool_6mA_plus = 722
length_bool_6mA_min = 663
#hdf5 file used 
HDF5_5mC='perread_Ship_Green_100plus_fullmethy.5mC.tombo.per_read_stats'
HDF5_6mA = 'perread_Ship_Green_100plus_fullmethy.6mA.tombo.per_read_stats'

#collect 5mC sites from other file
data_plus,data_min,DNA_arr_list = GL_5mC.processing_data(HDF5_5mC,chrom,block_stat,x_min,x_max,length_bool_5mC_plus,length_bool_5mC_min,data_size,methy_threshold)
#collect 6mA sites from other file
#df_sorted_plus,df_sorted_min,DNA_arr_list = GL_6mA.processing_6mA_data(HDF5_6mA,chrom,block_stat,x_min,x_max,length_bool_6mA_plus,length_bool_6mA_min)
#Tool to calculate expected nucleosome occupancy used as reference
#N = Nucleosome_TOOL_calculator(DNA_arr_list,x_min,x_max)
#combines the 5mC and 6mA sites into one dataframe
#methyl_plus,methyl_min,plus_cutoff = combiner_5mC_6mA(data_plus,data_min,df_sorted_plus,df_sorted_min)
#combines both plus and minus strand
#methyl_tot_data = pd.concat([methyl_plus,methyl_min],axis=1)
#plots the methylation and NUS for every read and the NUS
#output is the NUS data and the mean NUS
#PCA_nucleo_df_plus,nucl_occup_mean_plus = GL_6mA.nucleosome_density(methyl_plus,x_min,x_max,N,methyl_plus.shape[1],TSS,1,0.80,'GAL3','up','GAL3_again_plus')
#PCA_nucleo_df_min, nucl_occup_mean_min = GL_6mA.nucleosome_density(methyl_min,x_min,x_max,N,0,TSS,1,0.8,'GAL3','up','GAL3_again_min')




