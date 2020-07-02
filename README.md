# Bachelor Research Project: Nucleosome mapping in single long DNA molecules detected by nanopore methylation sequencing
# Twan Hooijschuur
All the python script used in my Bachelor Research Project. 

Tombo_methylation_map_GAL1_10_and_4x601.py is used to map methylation in the first experiment for testing methylation on the 4x601 and GAL1_10 plasmids. The tool can't map nucleosomes. It is possible to map different methylation fractions with a CpG LLR against GpC LLR plot. There is also a function to compare probability denstities of different methylation fractions

Shipony_command_file.py is the basefile used to map full yeast genome reads from Shipony et al. 2020. It uses functions of the two other Shipony scripts, which extract 5mC or 6mA methylation form the HDF5 per read statistics file.

GAL_1_10_16x610_methylation_map_and_NUS.py is the file designed to read plasmid stastics files where you can adjust the parameters to fit you plasmid. It is designed to center you reads around a cutting site. It makes a methylation map and maps the NUS for single reads. It is more flexible to use for new plasmid nanopore reads

NucleosomePositionCore.py is a file from the van der Heijden et al. "Sequence-based prediction of single nucleosome positioning and genome-wide nucleosome occupancy." PNAS (2013): 6240-6240. It is used to plot expected nucleosome position and was used in the case no MNase reference was available to compare with.It's use is not essential and for more details see the tool on: http://bio.physics.leidenuniv.nl/~noort/cgi-bin/nup3_st.py
