# crcPIV
Processing of PIV results at CRC-LETE lab using python

crcPIV-pre.py
  - Objectives:
    1. Read time sequence tecplot data (vector fields in .dat format) and generate .npy files. This step uses large amount of time and RAM.
    2. Run turbulence/statistics calculations.
    3. Run uncertainty calculations.
    4. Run any other analysis that require all timesteps.
    5. Generate .npy file with processed/reduced results (mean, variance, turbulence fields,uncertainties etc).

crcPIV-pos.py
  - Objectives:
    1. Read processed/reduced results in .npy files. This step uses much less memory and make it possible to study several cases simultaneously.
    2. Run calculations that do not require time information.
    3. Produce plots and line data graphs. This step may be used for comparison with external data (anemometry/CFD) or several PIV cases.

To use only run crcPIV-(pre/pos).py in a terminal 'python crcPIV-pos.py'. It is advisable to make copies of crcPIV-(pre/pos).py file for each data set you want to process so not to erase the original file from repository and also better organize the processing steps for each case. Each copy can be placed in any directory.

EVTK package from https://bitbucket.org/pauloh/pyevtk
