import R2DGC as r
import os
import pandas as pd
import numpy as np

# Get in the correct directory
TICDir = '/Users/Carl/Desktop/R2DGC/R2DGC_QuickStart/GMB_processed_TIC/'
os.chdir(TICDir)
os.chdir('..')
if 'processed' not in os.listdir(): os.mkdir('processed')
if 'FAME' not in os.listdir(): os.mkdir('FAME')
os.chdir(TICDir)
inputFileList = os.listdir()
if '.DS_Store' in inputFileList: inputFileList.remove('.DS_Store')
#inputFileList = inputFileList[0:4]

# Input file for problem ions
inputFile = inputFileList[0]

# Find common ions
print('Finding problem ions...')
ProblemIons = r.FindProblemIons(inputFile)
commonIons = ProblemIons.loc[:,'Ions']
#commonIons = []

# # Precompress files
print('Precompressing files...')
PeaksToCompress = r.PrecompressFiles(inputFileList=inputFileList,quantMethod="T",commonIons=commonIons,outputFiles=True)

# Find FAMEs
print('Finding FAME standards...')
os.chdir(TICDir)
FAME_Frame = '../FIND_FAME_FRAME.txt'
os.chdir('../processed')
inputFileList = os.listdir()
if '.DS_Store' in inputFileList: inputFileList.remove('.DS_Store')
r.Find_FAME_Standards(inputFileList,FAME_Frame)

# Standards
RT1_Standards = pd.Series(['FAME_8', 'FAME_10', 'FAME_12', 'FAME_14', 'FAME_16', 'FAME_18', 'FAME_20', 'FAME_22', 'FAME_24'])
RT2_Standards = pd.Series(['FAME_8', 'FAME_10', 'FAME_12', 'FAME_14', 'FAME_16', 'FAME_18', 'FAME_20', 'FAME_22', 'FAME_24'])

# Align peaks
print('Aligning peaks...')
os.chdir(TICDir)
os.chdir('../FAME')
inputFileList = os.listdir()
if '.DS_Store' in inputFileList: inputFileList.remove('.DS_Store')
Alignment = r.ConsensusAlign(inputFileList,RT1_Standards=RT1_Standards,RT2_Standards=RT2_Standards,commonIons=commonIons)
AlignmentTable = Alignment[0].iloc[0]

# Write out alignment table
os.chdir('..')
AlignmentTable.to_csv('AlignmentTable.txt')

print('Alignment table shape:')
print(AlignmentTable.shape)
print('Missing values:')
print(AlignmentTable.iloc[:,1:].isnull().values.sum()/((AlignmentTable.shape[1]-1)*AlignmentTable.shape[0]))
