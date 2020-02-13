# HealthInf2020


User Manual

There are two applications called erders.py and unn.py. Both applications have been developed and tested in python version 3.6.8 therefore, it is appropriate to run applications in this version of python. You must also have the needed packages installed.For the erders.py file, these are the "numpy" and "mne" packages. For the unn.py file, these are the "numpy", "pandas", and "keras" packages.
After all these things have been installed, you have to replace "EEG\\" with your data path in erders.py on line 66.
The next step is open the command line at the location of the application and enter command "python erders.py".
After this program is finished, two files (train.txt and test.txt) will be created. Then you have to rename these two files to train.csv and test.csv. Unfortunately, it was not possible to save the files directly in csv format because when it was saved in csv format it wrote nonsense commas and this problem could not be resolved.
You can start the classifier with command "python unn.py" now.
At the end of the classifier the minimum, maximum and average accuracy are displayed.
