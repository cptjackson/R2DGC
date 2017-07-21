import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import os
import copy
np.seterr(all='ignore')

def FindProblemIons(inputFile, possibleIons=pd.Series(np.arange(70,601,1)), absentIonThreshold=0.01, commonIonThreshold=2):

  #Read in and format input file
  currentRawFile = pd.read_table(inputFile, sep='\t', encoding='latin-1')
  currentRawFile = currentRawFile.dropna(axis=0,how='any')
  currentRawFile.iloc[:,3] = currentRawFile.iloc[:,3].apply(lambda row: 'TRUE' if row == 'T' else 'FALSE')
  #currentRawFile<-read.table(inputFile, header=T, sep="\t", fill=T, quote="",strip.white = T, stringsAsFactors = F)

  #Parse retention times
  RTSplit1 = currentRawFile.iloc[:,1].str.split(' , ').str[0]
  RTSplit2 = currentRawFile.iloc[:,1].str.split(' , ').str[1]
  currentRawFile["RT1"] = pd.to_numeric(RTSplit1)
  currentRawFile["RT2"] = pd.to_numeric(RTSplit2)

  #Remove duplicate metabolites
  uniqueIndex = currentRawFile.iloc[:,0] + currentRawFile.iloc[:,1] + currentRawFile.iloc[:,2].astype(str)
  currentRawFile = currentRawFile[uniqueIndex.duplicated() != True]
  currentRawFile.index = pd.RangeIndex(start=1, stop=currentRawFile.shape[0]+1, step=1)

  #Parse ion spectra column into list of numeric vectors
  spectraCol = currentRawFile.iloc[:,4]
  spectraSplit = spectraCol.str.split(' ')
  spectraSplit = spectraSplit.apply(lambda row: [i.split(':') for i in row])
  spectraSplit = spectraSplit.apply(lambda row: [[float(x) for x in i] for i in row])
  spectraSplit = spectraSplit.apply(lambda row: [[i[0] for i in row], [i[1] for i in row]])
  spectraSplit = spectraSplit.apply(lambda row: sorted(enumerate(row[1]), key=lambda x: row[0][x[0]]))
  spectraSplit = spectraSplit.apply(lambda row: [i[1] for i in row])

  #Identify absent ions
  spectraFrame = pd.DataFrame(spectraSplit)
  spectraFrame = spectraFrame['Spectra'].apply(pd.Series).T
  spectraFrame = spectraFrame.rename(columns = lambda x : 'V' + str(x))
  spectraSums = spectraFrame.sum(0)
  ionProportions = spectraFrame.T.apply(lambda x: sum((x/spectraSums)>absentIonThreshold))
  AbsentIons = possibleIons[ionProportions==0]

  #Calculate pairwise retention time comparisons
  RT1Index = currentRawFile["RT1"].apply(lambda x: abs(x-currentRawFile["RT1"]))
  RT2Index = currentRawFile["RT2"].apply(lambda x: abs(x-currentRawFile["RT2"])*100)

  #Calculate change global metabolite spectral similarity after dropping each ion not previously identified as an absent ion
  def CalcSum50(Ion,AbsentIons,possibleIons,spectraSplit):
    CommonIons = pd.Series(Ion).append(AbsentIons)
    indices = [ind for ind, item in enumerate(np.logical_not(possibleIons.isin(CommonIons))) if item == True]
    spectraSplitMask = spectraSplit.apply(lambda d: [d[idx] for idx in indices])
    spectraFrame = pd.DataFrame(spectraSplitMask)
    spectraFrame = spectraFrame['Spectra'].apply(pd.Series)
    spectraFrame = spectraFrame.T/np.sqrt(np.square(spectraFrame.T).apply(lambda row: sum(row)))
    SimilarityMatrix = spectraFrame.T.dot(spectraFrame)*100
    SimilarityMatrix = SimilarityMatrix-RT1Index-RT2Index
    SimilarityMatrix = np.triu(SimilarityMatrix,k=1)
    return(np.sum(np.sum(SimilarityMatrix>50,axis=1)))

  Sum50 = possibleIons[possibleIons.isin(AbsentIons) == False].apply(lambda x: CalcSum50(x, AbsentIons, possibleIons,spectraSplit))
  Sum50.index = possibleIons[possibleIons.isin(AbsentIons) == False]
  ProblemIons = possibleIons[possibleIons.isin(AbsentIons) == False][scale(Sum50)<(-1*commonIonThreshold)]

  #Plot commmon ion results
  #if(plotData==True):
#    graphics::plot(possibleIons[!possibleIons%in%AbsentIons],scale(Sum50), pch=16,cex=0.5, ylab="Sum 50 (Std. Dev.)", xlab="Ion")
#    if(length(ProblemIons)>0):
#      text(ProblemIons,scale(Sum50)[which(scale(Sum50)<(-2))], labels = ProblemIons, pos=3, cex=0.75, col="red")

  #Combine absent ions and common ions into a dataframe to output
  ionsToRemove = possibleIons[possibleIons.isin(AbsentIons)|possibleIons.isin(ProblemIons)]
  resultFrame = pd.DataFrame(data=possibleIons,columns=["Ions"])
  resultFrame['Status'] = pd.Series(['NA']*resultFrame.shape[0], index=resultFrame.index)
  indices = [ind for ind, item in enumerate(resultFrame.iloc[:,0].isin(AbsentIons)) if item == True]
  resultFrame.iloc[indices,1] = "Absent"
  indices = [ind for ind, item in enumerate(resultFrame.iloc[:,0].isin(ProblemIons)) if item == True]
  resultFrame.iloc[indices,1] = "Common"
  indices = [ind for ind, item in enumerate(resultFrame.iloc[:,1] == "NA") if item == False]
  return resultFrame.iloc[indices,:]


def PrecompressFiles(inputFileList, RT1Penalty=1, RT2Penalty=10,similarityCutoff=95, commonIons=[], quantMethod="T", outputFiles=False):

  #Create empty list to store record of all peaks that should be combined
  combinedList = pd.DataFrame()
  combinedList2 = pd.DataFrame()
  combinedFrame = pd.DataFrame()

  def ImportFile(inputFile,commonIons=[],quantMethod='T'):

    #Read in file
    print('Importing ' + inputFile + '...')
    #currentRawFile<-read.table(File, sep="\t", fill=T, quote="",strip.white = T, stringsAsFactors = F,header=T)
    currentRawFile = pd.read_table(inputFile, sep='\t', encoding='latin-1')
    currentRawFile = currentRawFile.dropna(axis=0,how='any')
    if quantMethod == 'T':
        currentRawFile.iloc[:,3] = currentRawFile.iloc[:,3].apply(lambda row: 'TRUE' if row == 'T' else 'FALSE')

    #Parse retention times
    RTSplit1 = currentRawFile.iloc[:,1].str.split(' , ').str[0]
    RTSplit2 = currentRawFile.iloc[:,1].str.split(' , ').str[1]
    currentRawFile["RT1"] = pd.to_numeric(RTSplit1)
    currentRawFile["RT2"] = pd.to_numeric(RTSplit2)

    #Remove identical metabolite rows
    uniqueIndex = currentRawFile.iloc[:,0] + currentRawFile.iloc[:,1] + currentRawFile.iloc[:,2].astype(str)
    currentRawFile = currentRawFile[uniqueIndex.duplicated() != True]
    currentRawFile.index = pd.RangeIndex(start=1, stop=currentRawFile.shape[0]+1, step=1)

    #Parse metabolite spectra into a list
    spectraCol = currentRawFile['Spectra']
    spectraSplit = spectraCol.str.split(' ')
    spectraSplit = spectraSplit.apply(lambda row: [i.split(':') for i in row])
    spectraSplit = spectraSplit.apply(lambda row: [[float(x) for x in i] for i in row])
    spectraSplit = spectraSplit.apply(lambda row: [[i[0] for i in row], [i[1] for i in row]])
    spectraSplit = spectraSplit.apply(lambda row: [pd.Series(row[0])[~pd.Series(row[0]).isin(commonIons)].tolist(),pd.Series(row[1])[~pd.Series(row[0]).isin(commonIons)].tolist()])
    ionNames = sorted(spectraSplit[1][0])
    spectraSplit = spectraSplit.apply(lambda row: sorted(enumerate(row[1]), key=lambda x: row[0][x[0]]))
    spectraSplit = spectraSplit.apply(lambda row: [i[1] for i in row])
    return [currentRawFile,spectraSplit,ionNames]

  importedFiles = []
  [importedFiles.append(ImportFile(sample,commonIons)) for sample in inputFileList]

  #Calculate pair wise similarity scores between all metabolite spectras
  def FindMatches(Sample, RT1Penalty=1, RT2Penalty=10, similarityCutoff=95):

    spectraFrame = pd.DataFrame(Sample[1])
    spectraFrame = spectraFrame['Spectra'].apply(pd.Series)
    spectraFrame = spectraFrame.T/np.sqrt(np.square(spectraFrame.T).apply(lambda row: sum(row)))
    SimilarityMatrix = spectraFrame.T.dot(spectraFrame)*100


    #Subtract retention time difference penalties from similarity scores
    RT1Index = Sample[0]['RT1'].apply(lambda x: abs(x-Sample[0]['RT1'])*RT1Penalty)
    RT2Index = Sample[0]['RT2'].apply(lambda x: abs(x-Sample[0]['RT2'])*RT2Penalty)
    SimilarityMatrix = SimilarityMatrix-RT1Index-RT2Index
    np.fill_diagonal(SimilarityMatrix.values,0)

    #Find metabolites to with similarity scores greater than similarityCutoff to combine
    return SimilarityMatrix.apply(lambda x: x[x>=similarityCutoff].index.tolist())

  matchList = []
  print('Finding matches...')
  [matchList.append(FindMatches(sample,RT1Penalty, RT2Penalty, similarityCutoff)) for sample in importedFiles]

  #Initialize number of times to loop through match list in case more than two peaks need to be combined
  for sampNum,sample in enumerate(matchList):
    print('Analysing sample ' + str(sampNum+1) + '...')
    numReps = 0
    if len(sample.any()) > 0:
      numReps = max(sample.apply(lambda x: len(x)))-1

      if quantMethod == "U":

        #Find mates to combine
        Mates = sample.apply(lambda x: x[0] if x else [])
        MatesNotNaN = [val-1 for ind,val in enumerate(Mates) if val]
        MatesIndices = [ind for ind,val in enumerate(Mates) if val]
        NotMates = [ind for ind,val in enumerate(Mates) if not val]

        BindingQMs = importedFiles[sampNum][0].iloc[MatesNotNaN,3]
        BindingAreas = importedFiles[sampNum][0].iloc[MatesNotNaN,2]
        BindingSpectra = importedFiles[sampNum][1].iloc[MatesNotNaN]
        ionNames = pd.Series(importedFiles[sampNum][2])

        #Find mate partner to combine
        toBind = importedFiles[sampNum][0].iloc[MatesIndices,:]

        #Add peak info to combinedList to for output
        combinedList[inputFileList[sampNum]] = [toBind, importedFiles[sampNum][0].iloc[MatesNotNaN,:], inputFileList[sampNum]]
        toBind = toBind.assign(Bound=pd.Series(['NA']*toBind.shape[0], index=toBind.index))
        toBindQMs = toBind.iloc[:,3]
        toBindSpectra = importedFiles[sampNum][1].iloc[MatesIndices]
        toBindSpectra = toBindSpectra.apply(lambda x: pd.Series(x))

        #Perform proportional conversion to adjust peak areas with differing unique masses
        ConvNumerator = np.array([int(toBindSpectra.iloc[ind][ionNames==toBindQMs.iloc[ind]]) for ind,val in enumerate(toBindQMs)]).astype(float)
        ConvDenominator = np.array([int(toBindSpectra.iloc[ind][ionNames==BindingQMs.iloc[ind]]) for ind,val in enumerate(BindingQMs)]).astype(float)
        ConvDenominator[ConvDenominator==0] = np.nan
        BindingAreas.index = toBind.iloc[:,2].index
        toBind.loc[:,'Area'] = (toBind.iloc[:,2]*(ConvNumerator/ConvDenominator))+BindingAreas

        #Make sure only one combination (mate to partner) is included in output dataframe
        row1 = [val for val in Mates if val]
        row2 = [val+1 for val in MatesIndices if val]
        arr = np.array([row1,row2])
        minArr = np.apply_along_axis(lambda x: min(x),0,arr).astype(str)
        toBind.loc[:,'Bound'] = [item + "_" + minArr[ind] for ind,item in enumerate(toBind['Bound'])]
        toBind = toBind[toBind['Bound'].duplicated() != True]

        #Modify sample metabolite frame to include only combined peak
        importedFiles[sampNum][0] = importedFiles[sampNum][0].iloc[NotMates,:]
        importedFiles[sampNum][0] = importedFiles[sampNum][0].append(toBind.iloc[:,0:toBind.shape[1]-1])


      if quantMethod == "A" or quantMethod == "T":

        #Find mates to combine
        Mates = [val[0]-1 for ind,val in enumerate(sample) if val]
        NotMates = [ind for ind,val in enumerate(sample) if not val]
        MatesIndices = [ind for ind,val in enumerate(sample) if val]
        BindingAreas = importedFiles[sampNum][0].iloc[Mates,2]

        #Find mates partners to combine
        toBind = importedFiles[sampNum][0].iloc[MatesIndices,:]
        combinedList[inputFileList[sampNum]] = copy.deepcopy([toBind,importedFiles[sampNum][0].iloc[Mates,:]])
        combinedList[inputFileList[sampNum]][0].index = pd.RangeIndex(1,combinedList[inputFileList[sampNum]][0].shape[0]+1)
        combinedList[inputFileList[sampNum]][1].index = pd.RangeIndex(1,combinedList[inputFileList[sampNum]][1].shape[0]+1)
        toBind = toBind.assign(Bound=pd.Series(['NA']*toBind.shape[0], index=toBind.index))

        #Sum peak areas
        toBind.iloc[:,2] = toBind.iloc[:,2] + BindingAreas.values.tolist()

        #Ensure only one peak combination gets included in output
        row1 = [val for val in Mates]
        row2 = [val for val in MatesIndices]
        arr = np.array([row1,row2])
        minArr = np.apply_along_axis(lambda x: min(x),0,arr).astype(str)
        toBind.loc[:,'Bound'] = [item + "_" + minArr[ind] for ind,item in enumerate(toBind['Bound'])]
        toBind = toBind[toBind['Bound'].duplicated() != True]

        #Update sample metabolite file to include on combined peak
        importedFiles[sampNum][0] = importedFiles[sampNum][0].iloc[NotMates,:]
        importedFiles[sampNum][0] = importedFiles[sampNum][0].append(toBind.iloc[:,0:toBind.shape[1]-1])



    #If any metabolites had greater than two peaks to combine, loop through and make those combinations iteratively
    if numReps > 0:
      for rep in range(1,numReps+1):

        #Repeat similarity scores with combined peaks
        spectraCol = importedFiles[sampNum][0].iloc[:,4]
        spectraSplit = spectraCol.str.split(' ')
        spectraSplit = spectraSplit.apply(lambda row: [i.split(':') for i in row])
        spectraSplit = spectraSplit.apply(lambda row: [[float(x) for x in i] for i in row])
        spectraSplit = spectraSplit.apply(lambda row: [[i[0] for i in row], [i[1] for i in row]])
        spectraSplit = spectraSplit.apply(lambda row: [pd.Series(row[0])[~pd.Series(row[0]).isin(commonIons)].tolist(),pd.Series(row[1])[~pd.Series(row[0]).isin(commonIons)].tolist()])
        spectraSplit = spectraSplit.apply(lambda row: sorted(enumerate(row[1]), key=lambda x: row[0][x[0]]))
        spectraSplit = spectraSplit.apply(lambda row: [i[1] for i in row])

        spectraFrame = pd.DataFrame(spectraSplit)
        spectraFrame = spectraFrame['Spectra'].apply(pd.Series)
        spectraFrame = spectraFrame.T/np.sqrt(np.square(spectraFrame.T).apply(lambda row: sum(row)))
        SimilarityMatrix = spectraFrame.T.dot(spectraFrame)*100

        #Subtract retention time difference penalties from similarity scores
        RT1Index = importedFiles[sampNum][0]['RT1'].apply(lambda x: abs(x-importedFiles[sampNum][0]['RT1'])*RT1Penalty)
        RT2Index = importedFiles[sampNum][0]['RT2'].apply(lambda x: abs(x-importedFiles[sampNum][0]['RT2'])*RT2Penalty)
        SimilarityMatrix = SimilarityMatrix-RT1Index-RT2Index
        np.fill_diagonal(SimilarityMatrix.values,0)

        # Reindex
        SimilarityMatrix.index = pd.RangeIndex(1,SimilarityMatrix.shape[0]+1)
        SimilarityMatrix.columns = pd.RangeIndex(1,SimilarityMatrix.shape[0]+1)

        #Repeat peak combination if more combinations are necessary
        newMatchList = SimilarityMatrix.apply(lambda x: x[x>=similarityCutoff].index.tolist())

        if len(newMatchList.any()) > 0:
          if quantMethod == "U":

            #Find mates to combine
            Mates = newMatchList[sampNum].apply(lambda x: x[0] if x else [])
            MatesNotNaN = [val-1 for ind,val in enumerate(Mates) if val]
            MatesIndices = [ind for ind,val in enumerate(Mates) if val]
            NotMates = [ind for ind,val in enumerate(Mates) if not val]
            BindingQMs = importedFiles[sampNum][0].iloc[MatesNotNaN,3]
            BindingAreas = importedFiles[sampNum][0].iloc[MatesNotNaN,2]
            BindingSpectra = importedFiles[sampNum][1].iloc[MatesNotNaN]

            #Find mate partner to combine
            toBind = importedFiles[sampNum][0].iloc[MatesIndices,:]

            #Add peak info to combinedList to for output
            combinedList[inputFileList[sampNum]] = [toBind, importedFiles[sampNum][0].iloc[MatesNotNaN,:], inputFileList[sampNum]]
            toBind = toBind.assign(Bound=pd.Series(['NA']*toBind.shape[0], index=toBind.index))
            toBindQMs = toBind.iloc[:,3]

            toBindSpectra = importedFiles[sampNum][1].iloc[MatesIndices]
            toBindSpectra = toBindSpectra.apply(lambda x: pd.Series(x))

            #Perform proportional conversion to adjust peak areas with differing unique masses
            ConvNumerator = np.array([int(toBindSpectra.iloc[ind][ionNames==toBindQMs.iloc[ind]]) for ind,val in enumerate(toBindQMs)]).astype(float)
            ConvDenominator = np.array([int(toBindSpectra.iloc[ind][ionNames==BindingQMs.iloc[ind]]) for ind,val in enumerate(BindingQMs)]).astype(float)
            ConvDenominator[ConvDenominator==0] = np.nan
            BindingAreas.index = toBind.iloc[:,2].index
            toBind.loc[:,'Area'] = (toBind.iloc[:,2]*(ConvNumerator/ConvDenominator))+BindingAreas

            #Make sure only one combination (mate to partner) is included in output dataframe
            row1 = [val for val in Mates if len(val) > 0]
            row2 = [val+1 for val in MatesIndices if val]
            arr = np.array([row1,row2])
            minArr = np.apply_along_axis(lambda x: min(x),0,arr).astype(str)
            toBind.loc[:,'Bound'] = [item + "_" + minArr[ind] for ind,item in enumerate(toBind['Bound'])]
            toBind = toBind[toBind['Bound'].duplicated() != True]

            #Modify sample metabolite frame to include only combined peak
            importedFiles[sampNum][0] = importedFiles[sampNum][0].iloc[NotMates,:]
            importedFiles[sampNum][0] = importedFiles[sampNum][0].append(toBind.iloc[:,0:toBind.shape[1]-1])

          if quantMethod == "A" or quantMethod=="T":
            Mates = [val[0]-1 for ind,val in enumerate(newMatchList) if val]
            NotMates = [ind for ind,val in enumerate(newMatchList) if not val]
            MatesIndices = [ind for ind,val in enumerate(newMatchList) if val]
            BindingAreas = importedFiles[sampNum][0].iloc[Mates,2]

            toBind = importedFiles[sampNum][0].iloc[MatesIndices,:]
            toAdd = copy.deepcopy([toBind,importedFiles[sampNum][0].iloc[Mates,:]])
            toAdd[0].index = pd.RangeIndex(combinedList[inputFileList[sampNum]][0].shape[0]+1,combinedList[inputFileList[sampNum]][0].shape[0]+1+toAdd[0].shape[0])
            toAdd[1].index = pd.RangeIndex(combinedList[inputFileList[sampNum]][1].shape[0]+1,combinedList[inputFileList[sampNum]][1].shape[0]+1+toAdd[1].shape[0])
            combinedList[inputFileList[sampNum]][0] = pd.concat([combinedList[inputFileList[sampNum]][0],toAdd[0]])
            combinedList[inputFileList[sampNum]][1] = pd.concat([combinedList[inputFileList[sampNum]][1],toAdd[1]])

            toBind = toBind.assign(Bound=pd.Series(['NA']*toBind.shape[0], index=toBind.index))
            toBind.iloc[:,2] = toBind.iloc[:,2] + BindingAreas.values.tolist()

            row1 = [val for val in Mates]
            row2 = [val for val in MatesIndices]
            arr = np.array([row1,row2])
            minArr = np.apply_along_axis(lambda x: min(x),0,arr).astype(str)
            toBind.loc[:,'Bound'] = [item + "_" + minArr[ind] for ind,item in enumerate(toBind['Bound'])]
            toBind = toBind[toBind['Bound'].duplicated() != True]

            importedFiles[sampNum][0] = importedFiles[sampNum][0].iloc[NotMates,:]
            importedFiles[sampNum][0] = importedFiles[sampNum][0].append(toBind.iloc[:,0:toBind.shape[1]-1])




  #Make data frame with all combined peak pair info
  print('Combining files...')
  for fname in combinedList:
      nameList = pd.DataFrame([fname for i in range(1,combinedList[fname][1].shape[0]+1)],index=pd.RangeIndex(1,combinedList[fname][1].shape[0]+1),columns=['inputFileList[sampNum]'])
      combinedList2 = pd.concat([combinedList[fname][0],combinedList[fname][1],nameList],1)
      combinedFrame = pd.concat([combinedFrame,combinedList2])

  combinedFrame.index = pd.RangeIndex(1,combinedFrame.shape[0]+1)


  #If outputFiles==TRUE, write processed files out to the input file directory
  if outputFiles:
    print('Writing files...')
    os.chdir('../processed')
    for sampNum,sample in enumerate(importedFiles):
      sample[0].iloc[:,0:5].to_csv(inputFileList[sampNum][:-4] + "_Processed.txt",sep='\t',index=False)

  return combinedFrame


def Find_FAME_Standards(inputFileList, FAME_Frame="../FIND_FAME_FRAME.txt", numCores=4, RT1Penalty=1, RT2Penalty=10, similarityCutoffWarningThreshold=80):
  FAMES = pd.read_table(FAME_Frame,sep='\t',encoding='latin-1')
  RTSplit1 = FAMES.iloc[:,1].str.split(' , ').str[0]
  RTSplit2 = FAMES.iloc[:,1].str.split(' , ').str[1]
  FAMES["RT1"] = pd.to_numeric(RTSplit1)
  FAMES["RT2"] = pd.to_numeric(RTSplit2)

  FAMESspectraCol = FAMES.iloc[:,2]
  FAMESspectraSplit = FAMESspectraCol.str.split(' ')
  FAMESspectraSplit = FAMESspectraSplit.apply(lambda row: [i.split(':') for i in row])
  FAMESspectraSplit = FAMESspectraSplit.apply(lambda row: [[float(x) for x in i] for i in row])
  FAMESspectraSplit = FAMESspectraSplit.apply(lambda row: [[i[0] for i in row], [i[1] for i in row]])
  FAMESspectraSplit = FAMESspectraSplit.apply(lambda row: sorted(enumerate(row[1]), key=lambda x: row[0][x[0]]))
  FAMESspectraSplit = FAMESspectraSplit.apply(lambda row: [i[1] for i in row])
  FAMESspectraFrame = pd.DataFrame(FAMESspectraSplit)
  FAMESspectraFrame = FAMESspectraFrame['Spectra'].apply(pd.Series)
  FAMESspectraFrame = FAMESspectraFrame.T/np.sqrt(np.square(FAMESspectraFrame.T).apply(lambda row: sum(row)))


  def AnnotateFAMES(inputFile):
    currentRawFile = pd.read_table(inputFile, sep='\t', encoding='latin-1')
    currentRawFile = currentRawFile.dropna(axis=0,how='any')
    #currentRawFile<-read.table(File, sep="\t", fill=T, quote="",strip.white = T, stringsAsFactors = F,header=T)

    #Parse retention times
    RTSplit1 = currentRawFile.iloc[:,1].str.split(' , ').str[0]
    RTSplit2 = currentRawFile.iloc[:,1].str.split(' , ').str[1]
    currentRawFile["RT1"] = pd.to_numeric(RTSplit1)
    currentRawFile["RT2"] = pd.to_numeric(RTSplit2)

    #Remove identical metabolite rows
    uniqueIndex = currentRawFile.iloc[:,0] + currentRawFile.iloc[:,1] + currentRawFile.iloc[:,2].astype(str)
    currentRawFile = currentRawFile[uniqueIndex.duplicated() != True]
    currentRawFile.index = pd.RangeIndex(start=1, stop=currentRawFile.shape[0]+1, step=1)

    #Parse metabolite spectra into a list
    spectraCol = currentRawFile.iloc[:,4]
    spectraSplit = spectraCol.str.split(' ')
    spectraSplit = spectraSplit.apply(lambda row: [i.split(':') for i in row])
    spectraSplit = spectraSplit.apply(lambda row: [[float(x) for x in i] for i in row])
    spectraSplit = spectraSplit.apply(lambda row: [[i[0] for i in row], [i[1] for i in row]])
    spectraSplit = spectraSplit.apply(lambda row: sorted(enumerate(row[1]), key=lambda x: row[0][x[0]]))
    spectraSplit = spectraSplit.apply(lambda row: [i[1] for i in row])

    #CalcSims
    spectraFrame = pd.DataFrame(spectraSplit)
    spectraFrame = spectraFrame['Spectra'].apply(pd.Series)
    spectraFrame = spectraFrame.T/np.sqrt(np.square(spectraFrame.T).apply(lambda row: sum(row)))
    SimilarityMatrix = FAMESspectraFrame.T.dot(spectraFrame)*100

    #Calculate pairwise RT penalties for each current file metabolite and seed file metabolites
    RT1Index = currentRawFile['RT1'].apply(lambda x: abs(x-FAMES['RT1'])*RT1Penalty).T
    RT2Index = currentRawFile['RT2'].apply(lambda x: abs(x-FAMES['RT2'])*RT2Penalty).T
    SimilarityMatrix = SimilarityMatrix-RT1Index-RT2Index

    # TO BE DONE - NEED FILE THAT TRIGGERS THESE ERRORS
    #if sum(SimilarityMatrix.T.apply(lambda row: max(row)) < similarityCutoffWarningThreshold) > 1:
    #  print("Potential Problem Match: ", FAMES[which(apply(SimilarityMatrix,1,max)<similarityCutoffWarningThreshold),1], "  "))

    #if(sum(duplicated(apply(SimilarityMatrix,1,which.max)))>0):
    #  print("A FAME peak is missing in ",File))

    currentRawFile.iloc[SimilarityMatrix.T.apply(lambda row: np.argmax(row)-1),0] = FAMES.iloc[:,0].values
    os.chdir('../FAME')
    currentRawFile.iloc[:,0:5].to_csv(inputFile[:-4] + "_FAME_appended.txt",sep='\t',index=False)
    os.chdir('../processed')


  [AnnotateFAMES(inputFile) for inputFile in inputFileList]

def ConsensusAlign(inputFileList,
                         RT1_Standards=None,
                         RT2_Standards=None,
                         seedFile=0, #Change to 5 for test case reproducibility
                         RT1Penalty=1,
                         RT2Penalty=10, #1 for relatively unstable spectras
                         autoTuneMatchStringency=True,
                         similarityCutoff=90,
                         disimilarityCutoff=0,
                         commonIons=[],
                         missingValueLimit=0.75,
                         missingPeakFinderSimilarityLax=0.85,
                         quantMethod="T"):

  #Function to import files
  def ImportFile(File, commonIons=[], RT1_Standards=None, RT2_Standards=None):
    #Read in file
    print('Importing ' + File + '...')
    currentRawFile = pd.read_table(File, sep='\t', encoding='latin-1')
    currentRawFile = currentRawFile.dropna(axis=0,how='any')
    #currentRawFile<-read.table(File, sep="\t", fill=T, quote="",strip.white = T, stringsAsFactors = F,header=T)
    #currentRawFile[,5]<-as.character(currentRawFile[,5])
    #currentRawFile<-currentRawFile[which(!is.na(currentRawFile[,3])&nchar(currentRawFile[,5])!=0),]
    #currentRawFile[,2]<-as.character(currentRawFile[,2])

    #Parse retention times
    RTSplit1 = currentRawFile.iloc[:,1].str.split(' , ').str[0]
    RTSplit2 = currentRawFile.iloc[:,1].str.split(' , ').str[1]
    currentRawFile["RT1"] = pd.to_numeric(RTSplit1)
    currentRawFile["RT2"] = pd.to_numeric(RTSplit2)

    #Remove identical metabolite rows
    uniqueIndex = currentRawFile.iloc[:,0] + currentRawFile.iloc[:,1] + currentRawFile.iloc[:,2].astype(str)
    currentRawFile = currentRawFile[uniqueIndex.duplicated() != True]
    currentRawFile.index = pd.RangeIndex(start=1, stop=currentRawFile.shape[0]+1, step=1)

    #Index metabolites by RT1 Standards
    if RT1_Standards is not None:
      #Check if all RT1 standards are present in each file
      if sum(RT1_Standards.isin(currentRawFile.iloc[:,0])==True) != len(RT1_Standards):
        raise ValueError("Seed file missing RT1 standards:", RT1_Standards[~RT1_Standards.isin(currentRawFile.iloc[:,0])].values)

      #Index each metabolite by RT1 Standards
      RT1_Length = max(currentRawFile[currentRawFile.iloc[:,0].isin(RT1_Standards)].iloc[:,5])-min(currentRawFile[currentRawFile.iloc[:,0].isin(RT1_Standards)].iloc[:,5])
      for Standard in RT1_Standards:
        currentRawFile[Standard + "_RT1"] = (currentRawFile.iloc[:,5] - currentRawFile[currentRawFile.iloc[:,0] == Standard].iloc[0][5])/RT1_Length


    #Index metabolites by RT2 Standards
    if RT2_Standards is not None:
      #Check if all RT2_Standards are present
      if sum(RT2_Standards.isin(currentRawFile.iloc[:,0])==True) != len(RT2_Standards):
        raise ValueError("Seed file missing RT2 standards:", RT2_Standards[~RT2_Standards.isin(currentRawFile.iloc[:,0])].values)

      #Index each metabolite by RT2 standards
      RT2_Length = max(currentRawFile[currentRawFile.iloc[:,0].isin(RT2_Standards)].iloc[:,5])-min(currentRawFile[currentRawFile.iloc[:,0].isin(RT2_Standards)].iloc[:,5])
      for Standard in RT2_Standards:
        currentRawFile[Standard + "_RT2"] = (currentRawFile.iloc[:,5] - currentRawFile[currentRawFile.iloc[:,0] == Standard].iloc[0][5])/RT2_Length


    #Parse metabolite spectra into a list
    spectraCol = currentRawFile.iloc[:,4]
    spectraSplit = spectraCol.str.split(' ')
    spectraSplit = spectraSplit.apply(lambda row: [i.split(':') for i in row])
    spectraSplit = spectraSplit.apply(lambda row: [[float(x) for x in i] for i in row])
    spectraSplit = spectraSplit.apply(lambda row: [[i[0] for i in row], [i[1] for i in row]])
    spectraSplit = spectraSplit.apply(lambda row: [pd.Series(row[0])[~pd.Series(row[0]).isin(commonIons)].tolist(),pd.Series(row[1])[~pd.Series(row[0]).isin(commonIons)].tolist()])
    spectraSplit = spectraSplit.apply(lambda row: sorted(enumerate(row[1]), key=lambda x: row[0][x[0]]))
    spectraSplit = spectraSplit.apply(lambda row: [i[1] for i in row])

    return [currentRawFile,spectraSplit]

  ImportedFiles = []
  [ImportedFiles.append(ImportFile(inputFileList[ind],commonIons,RT1_Standards,RT2_Standards)) for ind,sample in enumerate(inputFileList)]

  # Remove None values
  ImportedFiles = [x for x in ImportedFiles if x is not None]

  #Function to calculate pair wise similarity scores between all metabolite spectras
  def GenerateSimFrames(Sample, SeedSample, RT1Penalty=1, RT2Penalty=10, RT1_Standards=None, RT2_Standards=None):
    seedSpectraFrame = pd.DataFrame(SeedSample[1])
    seedSpectraFrame = seedSpectraFrame['Spectra'].apply(pd.Series)
    seedSpectraFrame = seedSpectraFrame.T/np.sqrt(np.square(seedSpectraFrame.T).apply(lambda row: sum(row)))
    sampleSpectraFrame = pd.DataFrame(Sample[1])
    sampleSpectraFrame = sampleSpectraFrame['Spectra'].apply(pd.Series)
    sampleSpectraFrame = sampleSpectraFrame.T/np.sqrt(np.square(sampleSpectraFrame.T).apply(lambda row: sum(row)))
    SimilarityMatrix = seedSpectraFrame.T.dot(sampleSpectraFrame)*100

    #Calculate pairwise RT penalties for each current file metabolite and seed file metabolites
    RT1Index = Sample[0]['RT1'].apply(lambda x: abs(x-SeedSample[0]['RT1'])*RT1Penalty).T
    RT2Index = Sample[0]['RT2'].apply(lambda x: abs(x-SeedSample[0]['RT2'])*RT2Penalty).T

    #Use RT indices to calculate RT penalties if necessary
    if RT1_Standards is not None:
      #Compute list of metabolite to RT1 standard differences between current file and seed file for each metabolite
      RT1Index = []
      RT1_Length = max(Sample[0][Sample[0].iloc[:,0].isin(RT1_Standards)].iloc[:,5])-min(Sample[0][Sample[0].iloc[:,0].isin(RT1_Standards)].iloc[:,5])

      for Standard in RT1_Standards:
        RT1Index.append((Sample[0][Standard + '_RT1'].apply(lambda x: abs(x-SeedSample[0][Standard + '_RT1'])*RT1Penalty/len(RT1_Standards))*RT1_Length).T)

      #Sum all relative standard differences into a final score
      RT1Index = sum(RT1Index)

    if RT2_Standards is not None:
      #Compute list of metabolite to RT2 standard differences between current file and seed file for each metabolite
      RT2Index = []
      RT2_Length = max(Sample[0][Sample[0].iloc[:,0].isin(RT2_Standards)].iloc[:,5])-min(Sample[0][Sample[0].iloc[:,0].isin(RT2_Standards)].iloc[:,5])

      for Standard in RT2_Standards:
        RT2Index.append((Sample[0][Standard + '_RT2'].apply(lambda x: abs(x-SeedSample[0][Standard + '_RT2'])*RT2Penalty/len(RT2_Standards))*RT2_Length).T)

      #Sum all relative standard differences into a final score
      RT2Index = sum(RT2Index)

    return SimilarityMatrix-RT1Index-RT2Index

  AlignmentTableList = []

  if isinstance(seedFile, int):
    seedList = [seedFile]
  else:
    seedList = seedFile


  for seed in seedList:
    print("seed is",seed)
    SeedSample = copy.deepcopy(ImportedFiles[seed])

    Indices = [ind for ind,item in enumerate(ImportedFiles)]
    NonSeedIndices = Indices[:seed]+Indices[(seed+1):]

    #Calculate pairwise similarity scores
    SimCutoffs = pd.Series([GenerateSimFrames(x, SeedSample, RT1Penalty, RT2Penalty, RT1_Standards, RT2_Standards) for x in (ImportedFiles[:seed]+ImportedFiles[(seed+1):])])
    SimCutoffs.index = NonSeedIndices

    #Calculate optimal similarity score cutoff if desired
    if autoTuneMatchStringency:
      SimScores = [[np.float64(np.count_nonzero(np.sum(y>x,1)>0))/(np.count_nonzero(y>x)**0.5) for x in range(1,101)] for y in SimCutoffs]
      SimScores = pd.DataFrame(SimScores).T
      similarityCutoff = np.argmax(np.sum(SimScores,1)) + 1
      disimilarityCutoff = similarityCutoff - 90


    #Find Metabolites to add to seed file
    for SampNum in range(0,len(ImportedFiles)):
      if SampNum is not seed:
          #Find best matches and mate pairs for each metabolite and remove inferior matches if metabolite is matched twice
          MatchScores = SimCutoffs[SampNum].max()
          Mates = SimCutoffs[SampNum].idxmax()
          Mates = Mates.iloc[np.argsort(-MatchScores)]
          MatchScores = MatchScores.iloc[np.argsort(-MatchScores)]
          MatchScores[Mates.duplicated()] = np.nan
          Mates = Mates.sort_index()
          MatchScores = MatchScores.sort_index()

          #Find metabolites in current file sufficiently dissimilar to add to alignment matrix
          SeedSample[0] = SeedSample[0].append(ImportedFiles[SampNum][0][MatchScores<disimilarityCutoff])
          SeedSample[0].index = pd.RangeIndex(1,len(SeedSample[0])+1)

          if (MatchScores<disimilarityCutoff).any():
            SeedSample[1] = SeedSample[1].append(ImportedFiles[SampNum][1][MatchScores<disimilarityCutoff])
            SeedSample[1].index = pd.RangeIndex(1,len(SeedSample[1])+1)



    #Repeat pairwise similarity score calculation on full seed sample file
    SimCutoffs = pd.Series([GenerateSimFrames(x, SeedSample, RT1Penalty, RT2Penalty, RT1_Standards, RT2_Standards) for x in ImportedFiles])

    #Calculate optimal similarity score cutoff if desired
    if autoTuneMatchStringency:
      print("Computing peak similarity threshold")
      SimScores = [[np.float64(np.count_nonzero(np.sum(y>x,1)>0))/(np.count_nonzero(y>x)**0.5) for x in range(1,101)] for y in SimCutoffs[:seed].append(SimCutoffs[(seed+1):])]
      SimScores = pd.DataFrame(SimScores).T
      similarityCutoff = np.argmax(np.sum(SimScores,1))+1
      disimilarityCutoff = similarityCutoff - 90


    #Establish alignment matrix
    inputFileList.insert(0,'Sample')
    FinalMatrix = pd.DataFrame(index=pd.RangeIndex(1,len(SeedSample[0])+1,1),columns=inputFileList)
    FinalMatrix.iloc[:,0] = SeedSample[0].iloc[:,0]
    inputFileList = inputFileList[1:]

    #Establish emptly list to store incongruent quant matches if using quantMethod "U" or "A"
    MissingQMList = []

    #Loop back through input files and find matches above similarityCutoff threshold
    for SampNum in range(0,len(ImportedFiles)):
      #Find best seed sample match scores
      MatchScores = SimCutoffs[SampNum].max()

      #Find current sample metabolites with best match
      Mates = SimCutoffs[SampNum].idxmax()
      Mates = Mates.iloc[np.argsort(-MatchScores)]
      MatchScores = MatchScores.iloc[np.argsort(-MatchScores)]
      MatchScores[Mates.duplicated()] = np.nan
      Mates = Mates.sort_index()
      MatchScores = MatchScores.sort_index()

    #   if quantMethod == "U":
    #     #Find quant masses for each match pair
    #     MatchedSeedQMs = SeedSample[0]
    #     MatchedSeedQMs<- SeedSample[[1]][,4][Mates[which(MatchScores>=similarityCutoff)]]
    #     currentFileQMs<- ImportedFiles[[SampNum]][[1]][which(MatchScores>=similarityCutoff),4]
    #     #Add incongruent quant mass info to MissingQMList for output
    #     MissingQMList[[inputFileList[SampNum]]]<-cbind(inputFileList[SampNum],which(MatchScores>=similarityCutoff),currentFileQMs,inputFileList[seed],Mates[which(MatchScores>=similarityCutoff)],MatchedSeedQMs)[which(currentFileQMs!=MatchedSeedQMs),]
    #     #Convert areas proportionally for incongruent quant masses
    #     currentFileAreas<- ImportedFiles[[SampNum]][[1]][which(MatchScores>=similarityCutoff),3]
    #     currentFileSpectra<- ImportedFiles[[SampNum]][[2]][which(MatchScores>=similarityCutoff)]
    #     MatchedSeedSpectra<- SeedSample[[2]][Mates[which(MatchScores>=similarityCutoff)]]
    #     ConvNumerator<-unlist(lapply(1:length(currentFileQMs), function(x) currentFileSpectra[[x]][which(currentFileSpectra[[x]][,1]==currentFileQMs[x]),2]))
    #     ConvDenominator<-unlist(lapply(1:length(currentFileQMs), function(x) currentFileSpectra[[x]][which(currentFileSpectra[[x]][,1]==MatchedSeedQMs[x]),2]))
    #     ConvDenominator[which(ConvDenominator==0)]<-NA
    #     #Add matched peaks to final alignment matrix
    #     FinalMatrix[Mates[which(MatchScores>=similarityCutoff)],inputFileList[SampNum]]<-currentFileAreas*(ConvNumerator/ConvDenominator)
    #   #
    #   #
    #   if quantMethod == "A":
    #     #Make function to parse apexing masses and test whether 50% are in common with seed file
    #     def TestQMOverlap(x):
    #       SeedQMs<- strsplit(x[1],"\\+")
    #       FileQMs<- strsplit(x[2],"\\+")
    #       sum(unlist(SeedQMs)%in%unlist(FileQMs))/min(length(unlist(SeedQMs)),length(unlist(FileQMs)))<0.5
      #
    #     #Test apexing mass overlap for each metabolite match
    #     MatchedSeedQMs<- SeedSample[[1]][,4][Mates[which(MatchScores>=similarityCutoff)]]
    #     currentFileQMs<- ImportedFiles[[SampNum]][[1]][which(MatchScores>=similarityCutoff),4]
    #     QM_Bind<-cbind(MatchedSeedQMs,currentFileQMs)
    #     QM_Match<-apply(QM_Bind, 1, function(x) TestQMOverlap(x))
    #     #Add incongruent apexing masses to MissingQMList for output
    #     MissingQMList[[inputFileList[SampNum]]]<-cbind(inputFileList[SampNum],which(MatchScores>=similarityCutoff),currentFileQMs,inputFileList[seed],Mates[which(MatchScores>=similarityCutoff)],MatchedSeedQMs)[which(QM_Match==TRUE),]
    #     #Add matched peaks to final alignment matrix
    #     FinalMatrix[Mates[which(MatchScores>=similarityCutoff)],inputFileList[SampNum]]<-ImportedFiles[[SampNum]][[1]][which(MatchScores>=similarityCutoff),3]


      if quantMethod == "T":
        #Add newly aligned peaks to alignment matrix
        results = ImportedFiles[SampNum][0].loc[MatchScores>=similarityCutoff,'Area']
        results.index = Mates[MatchScores>=similarityCutoff].values
        FinalMatrix.loc[:,inputFileList[SampNum]] = results



    #Filter final alignment matrix to only peaks passing the missing value limit
    SeedSample[0] = SeedSample[0][np.sum(np.isnan(FinalMatrix.iloc[:,1:]),1) <= round(len(inputFileList)*(1-missingValueLimit))]
    SeedSample[1] = SeedSample[1][np.sum(np.isnan(FinalMatrix.iloc[:,1:]),1) <= round(len(inputFileList)*(1-missingValueLimit))]
    FinalMatrix = FinalMatrix[np.sum(np.isnan(FinalMatrix.iloc[:,1:]),1) <= round(len(inputFileList)*(1-missingValueLimit))]
    FinalMatrix.index = pd.RangeIndex(1,len(FinalMatrix)+1,1)

    #Compute relaxed similarity cutoff with missingPeakFinderSimilarityLax
    similarityCutoff = similarityCutoff*missingPeakFinderSimilarityLax
    print("Searching for missing peaks")

    # Reset indices
    SeedSample[0].index = pd.RangeIndex(1,len(SeedSample[0])+1,1)
    SeedSample[1].index = pd.RangeIndex(1,len(SeedSample[1])+1,1)

    #Loop through each file again and check for matches in high probability missing metabolites meeting relaxed similarity cutoff
    SimCutoffs = pd.Series([GenerateSimFrames(x, SeedSample, RT1Penalty, RT2Penalty, RT1_Standards, RT2_Standards) for x in ImportedFiles])

    #Find peaks with missing values
    for SampNum in range(0,len(ImportedFiles)):
      MissingPeaks = pd.Series(np.where(np.isnan(FinalMatrix.loc[:,inputFileList[SampNum]])))

      if len(MissingPeaks[0]) > 0:

        #Find best seed sample match scores
        MatchScores = SimCutoffs[SampNum].max()

        #Find current sample metabolites with best match
        Mates = SimCutoffs[SampNum].idxmax()
        Mates = Mates.iloc[np.argsort(-MatchScores)]
        MatchScores = MatchScores.iloc[np.argsort(-MatchScores)]
        MatchScores[Mates.duplicated()] = np.nan
        Mates = Mates.sort_index()
        MatchScores = MatchScores.sort_index()
        MatchScores = MatchScores[Mates.isin(MissingPeaks[0]+1)]
        Mates = Mates[Mates.isin(MissingPeaks[0]+1)]

        #If matches are greater than relaxed simlarity cutoff add to final alignment table
        if len(MatchScores[MatchScores>=similarityCutoff].index) > 0:
        #   if quantMethod == "U":
        #     #Find quant masses for each match pair
        #     MatchedSeedQMs<- SeedSample[[1]][,4][Mates[which(MatchScores>=similarityCutoff)]]
        #     currentFileQMs<- ImportedFiles[[SampNum]][[1]][which(MatchScores>=similarityCutoff),4]
        #     #Add incongruent quant mass info to MissingQMList for output
        #     MissingQMList[[paste0(inputFileList[SampNum],"_MPF")]]<-cbind(inputFileList[SampNum],which(MatchScores>=similarityCutoff),currentFileQMs,inputFileList[seed],Mates[which(MatchScores>=similarityCutoff)],MatchedSeedQMs)[which(currentFileQMs!=MatchedSeedQMs),]
        #     #Convert areas proportionally for incongruent quant masses
        #     currentFileAreas<- ImportedFiles[[SampNum]][[1]][names(which(MatchScores>=similarityCutoff)),3]
        #     currentFileSpectra<- ImportedFiles[[SampNum]][[2]][names(which(MatchScores>=similarityCutoff))]
        #     MatchedSeedSpectra<- SeedSample[[2]][Mates[which(MatchScores>=similarityCutoff)]]
        #     ConvNumerator<-unlist(lapply(1:length(currentFileQMs), function(x) currentFileSpectra[[x]][which(currentFileSpectra[[x]][,1]==currentFileQMs[x]),2]))
        #     ConvDenominator<-unlist(lapply(1:length(currentFileQMs), function(x) currentFileSpectra[[x]][which(currentFileSpectra[[x]][,1]==MatchedSeedQMs[x]),2]))
        #     ConvDenominator[which(ConvDenominator==0)]<-NA
        #     #Add matched peaks to final alignment matrix
        #     FinalMatrix[Mates[which(MatchScores>=similarityCutoff)],inputFileList[SampNum]]<-currentFileAreas*(ConvNumerator/ConvDenominator)
          #
        #   if quantMethod == "A":
        #     #Make function to parse apexing masses and test whether 50% are in common with seed file
        #     def TestQMOverlap(x):
        #       SeedQMs<- strsplit(x[1],"\\+")
        #       FileQMs<- strsplit(x[2],"\\+")
        #       return(sum(unlist(SeedQMs)%in%unlist(FileQMs))/min(length(unlist(SeedQMs)),length(unlist(FileQMs)))<0.5)
          #
        #     #Test apexing mass overlap for each metabolite match
        #     MatchedSeedQMs<- SeedSample[[1]][,4][Mates[which(MatchScores>=similarityCutoff)]]
        #     currentFileQMs<- ImportedFiles[[SampNum]][[1]][which(MatchScores>=similarityCutoff),4]
        #     QM_Bind<-cbind(MatchedSeedQMs,currentFileQMs)
        #     QM_Match<-apply(QM_Bind, 1, function(x) TestQMOverlap(x))
        #     #Add incongruent apexing masses to MissingQMList for output
        #     MissingQMList[[paste0(inputFileList[SampNum],"_MPF")]]<-cbind(inputFileList[SampNum],which(MatchScores>=similarityCutoff),currentFileQMs,inputFileList[seed],Mates[which(MatchScores>=similarityCutoff)],MatchedSeedQMs)[which(currentFileQMs!=MatchedSeedQMs),]
        #     #Add matched peaks to final alignment matrix
        #     FinalMatrix[Mates[which(MatchScores>=similarityCutoff)],inputFileList[SampNum]]<-ImportedFiles[[SampNum]][[1]][names(which(MatchScores>=similarityCutoff)),3]

          if quantMethod == "T":
            #Add matched peaks to final alignment matrix
            results = ImportedFiles[SampNum][0].loc[MatchScores[MatchScores>=similarityCutoff].index,'Area']
            results.index = Mates[MatchScores>=similarityCutoff].values
            FinalMatrix.loc[results.index,inputFileList[SampNum]] = results


    #Make MissingQMList into dataframe for output
    MissingQMFrame = pd.DataFrame(MissingQMList)
    AlignmentTableList.append(FinalMatrix)


  #If only one seed file is provided just output alignment matrix
  if isinstance(seedFile, int):
    FinalMatrix = AlignmentTableList[0]


  #If multiple seed files provided find peaks with >50% overlap across all seed files
  #if len(seedFile) > 1:

    #Find all peaks with at >50% alignment overlap
    #ConsensusPeaks = []
    #ConsensusMatches = []
    #ConsensusPeaks<-list()
    #ConsensusMatches<-list()

    #for i,item in enumerate(AlignmentTableList):

    #    Overlaps = item.apply(lambda y: AlignmentTableList[len(AlignmentTableList)-1].apply(lambda x: [sum(x) for x in y]))
    #    print(Overlaps)

    #Overlaps<-apply(AlignmentTableList[[i]], 1, function(y) apply(AlignmentTableList[[length(AlignmentTableList)]], 1, function(x) sum(x%in%y, na.rm = T)))
    #   Indexes<-arrayInd(which(Overlaps>(length(inputFileList)/2)), dim(Overlaps))
    #   row.names(Indexes)<-Indexes[,1]
    #   ConsensusPeaks[[i]]<-Indexes[,1]
    #   ConsensusMatches[[i]]<-Indexes
    #
    # ConsensusPeaks<-Reduce(intersect, ConsensusPeaks)
    #
    # #Filter all alignments to consensus peaks (>50% overlap)
    # AlignmentTableListFilt = AlignmentTableList
    # for i,val in enumerate(AlignmentTableList):
    #   AlignmentTableListFilt[[i]]<-AlignmentTableListFilt[[i]][ConsensusMatches[[i]][as.character(ConsensusPeaks),2],inputFileList]
    #
    # AlignmentTableListFilt[[length(AlignmentTableListFilt)]]<-AlignmentTableListFilt[[length(AlignmentTableListFilt)]][ConsensusPeaks,inputFileList]
    #
    # #Find median value of peak areas across all alignments
    # AlignVector<-data.frame(lapply(AlignmentTableListFilt, as.vector))
    # FinalMatrix<-matrix(apply(AlignVector,1,function(x) stats::median(x, na.rm=T)), nrow = length(ConsensusPeaks))
    # colnames(FinalMatrix)<-inputFileList
    # row.names(FinalMatrix)<- row.names(AlignmentTableListFilt[[length(AlignmentTableListFilt)]])
    #
    # #Filter metabolite info file by consensus peaks
    # SeedSample[0] = SeedSample[0].iloc[]
    # SeedSample[[1]]<-SeedSample[[1]][ConsensusPeaks,]


  #Add metabolite IDs if standardLibrary is used
  # if standardLibrary is not None:
  #   print("Matching peaks to standard library")
  #
  #   #Parse seed file spectras
  #   peakCol = SeedSample[0].iloc[:,4]
  #   peakSpectraSplit = peakCol.str.split(' ')
  #   peakSpectraSplit = peakSpectraSplit.apply(lambda row: [i.split(':') for i in row])
  #   peakSpectraSplit = peakSpectraSplit.apply(lambda row: [[float(x) for x in i] for i in row])
  #   peakSpectraSplit = peakSpectraSplit.apply(lambda row: [[i[0] for i in row], [i[1] for i in row]])
  #   peakSpectraSplit = peakSpectraSplit.apply(lambda row: [pd.Series(row[0])[~pd.Series(row[0]).isin(commonIons)].tolist(),pd.Series(row[1])[~pd.Series(row[0]).isin(commonIons)].tolist()])
  #   peakSpectraSplit = peakSpectraSplit.apply(lambda row: sorted(enumerate(row[1]), key=lambda x: row[0][x[0]]))
  #   peakSpectraSplit = peakSpectraSplit.apply(lambda row: [i[1] for i in row])
  #   peakSpectraFrame = pd.DataFrame(peakSpectraSplit)
  #   peakSpectraFrame = peakSpectraFrame['Spectra'].apply(pd.Series)
  #   peakSpectraFrame = peakSpectraFrame.T/np.sqrt(np.square(peakSpectraFrame.T).apply(lambda row: sum(row)))
  #
  #   #Parse standard library spectras
  #   standardCol = standardLibrary.iloc[:,2]
  #   standardSpectraSplit = standardCol.str.split(' ')
  #   standardSpectraSplit = standardSpectraSplit.apply(lambda row: [i.split(':') for i in row])
  #   standardSpectraSplit = standardSpectraSplit.apply(lambda row: [[float(x) for x in i] for i in row])
  #   standardSpectraSplit = standardSpectraSplit.apply(lambda row: [[i[0] for i in row], [i[1] for i in row]])
  #   standardSpectraSplit = standardSpectraSplit.apply(lambda row: [pd.Series(row[0])[~pd.Series(row[0]).isin(commonIons)].tolist(),pd.Series(row[1])[~pd.Series(row[0]).isin(commonIons)].tolist()])
  #   standardSpectraSplit = standardSpectraSplit.apply(lambda row: sorted(enumerate(row[1]), key=lambda x: row[0][x[0]]))
  #   standardSpectraSplit = standardSpectraSplit.apply(lambda row: [i[1] for i in row])
  #   standardSpectraFrame = pd.DataFrame(standardSpectraSplit)
  #   standardSpectraFrame = standardSpectraFrame['Spectra'].apply(pd.Series)
  #   standardSpectraFrame = standardSpectraFrame.T/np.sqrt(np.square(standardSpectraFrame.T).apply(lambda row: sum(row)))
  #   SimilarityMatrix = standardSpectraFrame.T.dot(peakSpectraFrame)*100
  #
  #
  #   #Compute RT differences
  #   RT1Index = SeedSample[0]['RT1'].apply(lambda x: abs(x-standardLibrary.iloc[:,3])*RT1Penalty)
  #   RT2Index = SeedSample[0]['RT2'].apply(lambda x: abs(x-standardLibrary.iloc[:,4])*RT2Penalty)
  #
  #   #Use RT indices to calculate RT penalties if necessary
  #   if RT1_Standards is not None:
  #     #Compute list of metabolite to RT1 standard differences between current file and seed file for each metabolite
  #     RT1Index = []
  #     RT1_Length = max(SeedSample[0][SeedSample[0].iloc[:,0].isin(RT1_Standards)].iloc[:,5])-min(SeedSample[0][SeedSample[0].iloc[:,0].isin(RT1_Standards)].iloc[:,5])
  #
  #     for Standard in RT1_Standards:
  #       RT1Index.append((SeedSample[0][Standard + '_RT1'].apply(lambda x: abs(x-standardLibrary[Standard + '_RT1'])*RT1Penalty/len(RT1_Standards))*RT1_Length).T)
  #
  #     #Sum all relative standard differences into a final score
  #     RT1Index = sum(RT1Index)
  #
  #   if RT2_Standards is not None:
  #     #Compute list of metabolite to RT2 standard differences between current file and seed file for each metabolite
  #     RT2Index = []
  #     RT2_Length = max(SeedSample[0][SeedSample[0].iloc[:,0].isin(RT2_Standards)].iloc[:,5])-min(SeedSample[0][SeedSample[0].iloc[:,0].isin(RT2_Standards)].iloc[:,5])
  #
  #     for Standard in RT2_Standards:
  #       RT2Index.append((SeedSample[0][Standard + '_RT2'].apply(lambda x: abs(x-standardLibrary[Standard + '_RT2'])*RT2Penalty/len(RT2_Standards))*RT2_Length).T)
  #
  #     #Sum all relative standard differences into a final score
  #     RT2Index = sum(RT2Index)
  #
  #   #Subtract RT penalties from Similarity Scores
  #   SimilarityMatrix = SimilarityMatrix-RT1Index-RT2Index
  #   #SimilarityMatrix<-SimilarityMatrix-RT1Index-RT2Index
  #   #row.names(SimilarityMatrix)<-standardLibrary[,1]
  #
  #   #Append top three ID matches to each metabolite and scores to seedRaw for output
  #   #SeedSample[[1]]<-cbind(t(apply(SimilarityMatrix,2,function(x) paste(names(x[order(-x)])[1:3],round(x[order(-x)][1:3],2),sep="_"))),SeedSample[[1]])
  #   #colnames(SeedSample[[1]])<-c("Library_Match_1","Library_Match_2","Library_Match_3",colnames(SeedSample[[1]])[4:length(colnames(SeedSample[[1]]))])

  #returnList<-list(FinalMatrix, SeedSample[[1]], MissingQMFrame)
  returnList = pd.DataFrame([FinalMatrix, SeedSample[0], MissingQMFrame])
  #names(returnList)<-c("Alignment_Matrix","Peak_Info","Unmatched_Quant_Masses")
  return returnList


def MakeReference(inputFileList, RT1_Standards=None, RT2_Standards=None):

  #Create empty list to add RT-indexed standards
  StandardLibList = []

  #Create empty list for missing retention indices
  MissingRTIndices = pd.DataFrame()

  for File in inputFileList:

    #Read in file
    currentFile = pd.read_table(File, sep='\t', encoding='latin-1')
    currentRawFile = currentRawFile.dropna(axis=0,how='any')

    #Parse retention time
    RTSplit1 = currentFile.iloc[:,1].str.split(' , ').str[0]
    RTSplit2 = currentFile.iloc[:,1].str.split(' , ').str[1]
    currentFile["RT1"] = pd.to_numeric(RTSplit1)
    currentFile["RT2"] = pd.to_numeric(RTSplit2)


    #Find RT1 differences from all RT1 standards
    if RT1_Standards is not None:
      if sum(RT1_Standards.isin(currentFile.iloc[:,0])==True) != len(RT1_Standards):
        MissingRTIndices[File] = RT1_Standards[RT1_Standards.isin(currentFile.iloc[:,0])]
        continue


      RT1_Length = max(currentFile[currentFile.iloc[:,0].isin(RT1_Standards)].iloc[:,3])-min(currentFile[currentFile.iloc[:,0].isin(RT1_Standards)].iloc[:,3])
      for Standard in RT1_Standards:
        currentFile["RT1_" + Standard] = (currentFile.iloc[:,3] - currentFile[currentFile.iloc[:,0] == Standard].iloc[0][3])/RT1_Length


    #Find RT2 differences from all RT2 standards
    if RT2_Standards is not None:
      if sum(RT2_Standards.isin(currentFile.iloc[:,0])==True) != len(RT2_Standards):
        MissingRTIndices[File] = [MissingRTIndices[File],RT2_Standards[RT2_Standards.isin(currentFile.iloc[:,0])]]
        continue

      RT2_Length = max(currentFile[currentFile.iloc[:,0].isin(RT2_Standards)].iloc[:,3])-min(currentFile[currentFile.iloc[:,0].isin(RT2_Standards)].iloc[:,3])
      for Standard in RT2_Standards:
        currentFile["RT2_" + Standard] = (currentFile.iloc[:,3] - currentFile[currentFile.iloc[:,0] == Standard].iloc[0][3])/RT2_Length


    #Add indexed metabolite only to StandardLibList
    StandardLibList.append(currentFile[~currentFile.iloc[:,0].isin(RT1_Standards)])


  #Convert library to dataframe for output
  if len(MissingRTIndices) == 0:
    StandardLibrary = pd.DataFrame(StandardLibList)
    StandardLibrary.reset_index(drop=True)
    return StandardLibrary

  if len(MissingRTIndices) > 0:
    print("Error: Missing RT indices detected. See output list")
    return MissingRTIndices
