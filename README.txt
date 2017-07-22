Please note that this code is still slow and doesn't quite do everything yet.
I'll send updates as I improve on it. I'll be performing the following steps:

• Step 1: fully implement all functionality.
• Step 2: change up the data types to use exclusively numpy arrays.
• Step 3: parallelise the code.

To run any of the Python code, you will need to import the module, e.g.:

 import R2DGC as r

Put that at the top of your script. You should also import pandas and numpy.
See the QuickStart.py file for an example.

In your script you can then call each function directly from the module, e.g.:

 Alignment = r.ConsensusAlign(inputFileList)

Then to run the script, type:

$ python <scriptname.py>

Another note: this code assumes that there is a directory with some files, a
directory on the same level called 'processed', and another called 'FAME'. See
the Quickstart.py file for an example.


FindProblemIons
---------------

This code should work exactly the same as the R code, with the exception of
plotting. It takes the following arguments:

inputFile                         str
possibleIons                      Series, default pd.Series(np.arange(70,601,1))
absentIonThreshold                float, default 0.01
commonIonThreshold                int, default 2

Returns a list of problem ions.


PrecompressFiles
----------------

Also works the same as the R code. This should work for both TIC and UM data. It
takes the following arguments:

inputFileList                           list of strs
RT1Penalty                              int, default 1
RT2Penalty                              int, default 1
similarityCutoff                        int, default 95
commonIons                              list, default []
quantMethod                             str, default 'T'
outputFiles                             bool, default False

Note that this assumes that there is a directory called 'processed' one level
above the directory with your files. If outputFiles is True, it will write to
this directory.

Returns a list of files to be compressed.


Find_FAME_STANDARDS
-------------------

For this to work you need the FIND_FAME_FRAME.txt to be one level above your
files by default. You can also supply the path to this file. It takes the
following arguments:

inputFileList                           list of strs
FAME_Frame                              str, default "../FIND_FAME_FRAME.txt"
RT1Penalty                              int, default 1
RT2Penalty                              int, default 10
similarityCutoffWarningThreshold        int, default 80

This also assumes that there is a directory called 'FAME' one level above the
directory with your files. It will write to this directory.


ConsensusAlign
--------------

Partially implemented, for TIC files only at the moment. Also does not handle
multiple seed files and currently cannot use a standard library. It takes the
following arguments:

inputFileList                           list of strs
RT1_Standards                           list of strs, default None
RT2_Standards                           list of strs, default None
seedFile                                int, default 0
RT1Penalty                              int, default 1
RT2Penalty                              int, default 10
autoTuneMatchStringency                 bool, default True,
similarityCutoff                        int, default 90,
disimilarityCutoff                      int, default 0,
commonIons                              list, default []
missingValueLimit                       float, default 0.75
missingPeakFinderSimilarityLax          float, default 0.85
quantMethod                             str, default "T"

Returns an alignment table.


MakeReference
-------------

This should work the same way as the R code. It takes the following arguments:

inputFileList                           list of strs
RT1_Standards                           list of strs, default None
RT2_Standards                           list of strs, default None

Returns a standard library or a list of missing RT indices.


FeihnMatch
----------

Not yet implemented. Coming soon!
