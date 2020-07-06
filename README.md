# ModifiedHopfieldNetwork 
### A Neural Dynamics Analysis Tool for Networks of Real or Simulated Neurons
ModifiedHopfieldNetwork is a neural analysis tool that can be used to estimate mutual information between neurons in a real or simulated network, can compute the accuracy of stimulus prediction using maximum a posteriori probability (MAP) models trained on neural data with Hopfield Networks, or can conduct an analysis of changing functional connectivity through an experiment  by breaking the experiment into equally sized chunks and then conducting MI/MAP analyses. 
### To use:
Download the **hopfield_map_estimate.py** file and explore the **demo.py** for a full explanation of all the attributes and their default values for the ModifiedHopfieldNetwork object. **_IMPORTANT:_** Your data will most likely not be in the same format as the data used for this experiment. In order to use your own data, you will likely need to rewrite the binaryVecs function to create binary, time-binned neural data with 0 representing that a neuron did not fire in a time window, and 1 representing that a neuron did.

### Contact:
#### If there are any problems with the code, or you would like help rewriting the binaryVecs function properly, feel free to email me at mhess21@cmc.edu and I will get back to you as soon as possible! Happy training!
