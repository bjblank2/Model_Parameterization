Model_Parameterization: Python 3.8 

Description of project: Fits model parameters for combined cluster expansion / Ising model
			This is a modified version of the code used by NamHoon to fit paramiters for his cluster expansion work.
Descriptoin of Files:
	celib.py             - contains functions for creating predictor matrix (counts number of times each cluster appears in a given DFT simulation)
	clusters_A_all       - data file containing cluster rules for austenite DFT simulations. 
			       A cluster rule describes [ atomic species inculded in a cluster a, b, c Ni=0 Mn=1 In=2], [neighbor distances between atoms in a cluster ab, ac, bc], [type of cluster, spin (1) chemical (0)]
	clusters_M_all       - data file containing cluster rules for martensite DFT simulations. 
			       Cluster rules are explaind above.
	count2_with_spins.py - Contains a single function that reads in DFT data (NiMnIn_aust/mart_all) and cluster rules and formats it so 
			       for regression functions. Calls celib.py to count clusters on DFT data points (creates predictor matrix)
	data_check.py        - Contains functions that check for / quantify multicolinearity in the data / model
	fit_ex.py            - Fits model parameters via Random Forest, LASSO CV, Ridge CV, and Linear Regression. Plots the results of the fit
			       and outputs the parameters to Fit_summery
	Fit_summery.txt      - Summery of the entire fitting process. Contains: list of clusteres considered, list of all non-redundant DFT data points, Energy in eV/atom of each point, 
			       number of times each cluster appears in the DFT point (counts/atom), Results of Lasso fit, Results of Ridge fit, Results of LinReg fit
	main.py		     - Main function. Call to run the project
	mathkit.py	     - List of funciotns used to handle all potential permutaions of a cluster
	Ni2MnIn.jpg	     - Shows smallest Austenite and Martensite system simulated in DFT
	NiMnIn_aust_all      - Summery of all Austenite data taken from DFT
			       Each new DFT simulation is inicated by a # followed by atomic species
			       The next line lists: composition, Name of the DFT structure, Energy, phase, and lattice constants
			       The following lines show: atom index, atomic species, spin, x-y-z fractional coordinates 
	NiMnIn_mart_all      - Summery of all Martensite data taken from DFT
	vasp.py		     - Contains functions for reading/writing POSCAR files and creating neighbor distance lists
	pert_summery	     - You can ignore this...