To start the Earth system analysis with different temperature rate, the follow steps need to be considered:
1) Choose which temperature rates, which coupling strengths and the number of ensemble members in the program Main_temprate_earths_system.py
1) Choose one line from the Latin hypercube distributed initial conditions file (i.e. one line from the file "lhs_preparator/latin_sh_file_save.txt"). This will initiate the computation of the respective run. Remember to replace $SLURMNTASKS with something else
2) Result files are safed under the directory results/feedbacks and the respective network setup (there are nine possibilities: [--, -0, -+, 0-, 00, 0+, +-, +0, ++])
3) Go to the directory evaluations. The program bandwidth_testing.py can be used to test which filtering bandwidth works best for the analysis based on one ensemble member, and the program plot_EWS.py can be used to visualize resilience indicators using a certain bandwidth and detrending window
4) In the evaluations directory, and use the program EWS_significance.py to calculate the resilience indicators autocorrelation and variance and their respective p values based on the chosen bandwidths.
5) Use the program success.py to visualize the significance of all resilience indicators
6) Use the program comparison_kendalltau.py to visualize the differences between methods used by Boers et.al. (2021) and van der Bolt et.al (2021). 



