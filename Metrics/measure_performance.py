import json
from os import mkdir, listdir, path
from numpy import *
import math
import matplotlib.pyplot as plt

resultsDir = '../Results/'
maxScanpathLength = 30


def main():
    amountOfModels = listdir(resultsDir)
    fig, ax = plt.subplots()
    for modelFolderName in amountOfModels:
        modelResultsPath = resultsDir + modelFolderName + '/'
        datasetsPerModel = listdir(modelResultsPath)
        for datasetName in datasetsPerModel:
            with open(modelResultsPath + datasetName + '/Scanpaths.json', 'r') as fp:
                scanpaths = json.load(fp)
            fixationsUntilTargetFound = []
            for index in range(0, maxScanpathLength): # los índices irían en el eje X y los valores del índice correspondiente en el eje Y
                fixationsUntilTargetFound.append(0)
            for imageName in scanpaths.keys():
                scanpathInfo = scanpaths[imageName]

                if (len(scanpathInfo['X']) < maxScanpathLength + 1) and scanpathInfo['target_found'] == True: 
                    for index in range(len(scanpathInfo['X']) - 1, maxScanpathLength): #el gráfico tiene que ser acumulativo, si se encontró con 2 fijaciones, también con 3,4,etc.
                        fixationsUntilTargetFound[index] += 1
        
                cumulativePerformance = list(map(lambda x: float(x) / len(scanpaths.keys()), fixationsUntilTargetFound))
            label = datasetName
            ax.plot(range(1, maxScanpathLength + 1), cumulativePerformance, label = label)
    ax.legend()   
    plt.xlabel('Number of fixations')
    plt.ylabel('Cummulative performance')
    plt.show()

main()
