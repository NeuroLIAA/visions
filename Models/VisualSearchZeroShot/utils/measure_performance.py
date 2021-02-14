import json
from os import mkdir, listdir, path
from numpy import *
import math
import matplotlib.pyplot as plt

resultsDir = '../results/'
maxScanpathLength = 30
totalStimuli = 240

def main():
    jsonsToMeasure = listdir(resultsDir)
    fig, ax = plt.subplots()
    for jsonFileName in jsonsToMeasure:
        if not(jsonFileName.startswith('Scanpaths')):
            continue
        jsonFile = open(resultsDir + jsonFileName, 'r')
        jsonStructs = json.load(jsonFile)
    
        jsonFile.close()
        fixationsUntilTargetFound = []

        for index in range(0, maxScanpathLength): # los índices irían en el eje X y los valores del índice correspondiente en el eje Y
            fixationsUntilTargetFound.append(0) 
        for struct in jsonStructs:
            if (type(struct['X']) is not list): # Hay algunos scanpaths que en matlab se guardaron como int en lugar de una lista con un solo int
                struct['X'] = [struct['X']]                    

            if (len(struct['X']) < maxScanpathLength + 1): #tendría que fijarme que el campo "target found" esté en true pero en matlab no estaba ese campo y además en todos los casos encuentra el target (creo)
                for index in range(len(struct['X']) - 1, maxScanpathLength): #el gráfico tiene que ser acumulativo, si se encontró con 2 fijaciones, también con 3,4,etc.
                    fixationsUntilTargetFound[index] += 1
        
            cumulativePerformance = list(map(lambda x: float(x) / 240.0, fixationsUntilTargetFound))
        label = jsonFileName[:-5].split('_')[1]
        ax.plot(range(1, maxScanpathLength + 1), cumulativePerformance, label = label)
    ax.legend()   
    plt.xlabel('Number of fixations')
    plt.ylabel('Cummulative performance')
    plt.show()

main()
