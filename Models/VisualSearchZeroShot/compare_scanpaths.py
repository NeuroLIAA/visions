import json
from os import mkdir, listdir, path

resultsDir = 'results/'

def main():
    jsonPythonFile = open(resultsDir + 'scanpathspython.json', 'r')
    jsonMatlabFile = open(resultsDir + 'scanpaths.json', 'r')
    jsonPythonStructs = json.load(jsonPythonFile)
    jsonMatlabStructs = json.load(jsonMatlabFile)
    jsonMatlabFile.close()
    jsonPythonFile.close()
    differences = []
    differentLengthScanpathsBy3OrMore = 0
    differentLengthScanpathsBy2OrLess = 0
    sameLengthDifferentScanpaths = 0
    for pythonStruct in jsonPythonStructs:
        for matlabStruct in jsonMatlabStructs:
            if pythonStruct['image'][3:6] == matlabStruct['image'][0:3]: #me fijo que estoy comparando scanpaths de la misma imagen
                if type(matlabStruct['X']) != type(pythonStruct['X']): #hay algunos scanpaths que en matlab se guardaron como int en lugar de una lista con un solo int
                    matlabStruct['X'] = [matlabStruct['X']]
                    matlabStruct['Y'] = [matlabStruct['Y']]
                lengthDifference = abs(len(pythonStruct['X']) - len(matlabStruct['X'])) #comparo las longitudes, si son distintas ya hay algún problema
                xData = compareStructs(pythonStruct, matlabStruct, 'X', 'X', lengthDifference > 0)
                yData = compareStructs(pythonStruct, matlabStruct, 'Y', 'Y', lengthDifference > 0)
                if lengthDifference > 0:
                    if lengthDifference > 3:
                        differentLengthScanpathsBy3OrMore+=1
                    else:                        
                        differentLengthScanpathsBy2OrLess+=1                        
                else:
                    sameLengthDifferentScanpaths+=xData['different paths?']
                differences.append({ "image" : pythonStruct['image'], "length difference" :lengthDifference, "X Python" : xData['python coords'], "X Matlab" : xData['matlab coords'], "Y Python" : yData['python coords'], "Y Matlab" : yData['matlab coords'], "fixations X distance" : xData['coords distance'], "fixations Y distance" : yData['coords distance']})
    differences.append({"scanpaths with different lengths by 3 or more fixations" : differentLengthScanpathsBy3OrMore, "scanpaths with different lengths by 2 or less fixations" : differentLengthScanpathsBy2OrLess, "scanpaths with same length but different paths" : sameLengthDifferentScanpaths})
    jsonDifferencesFile = open(resultsDir + 'scanpathsDifferences.json', 'w')
    json.dump(differences, jsonDifferencesFile, indent = 4)
    jsonDifferencesFile.close()




def compareStructs(firstStruct, secondStruct, fieldFirstStruct, fieldSecondStruct, isNotSameLength):
    inPython = list(map(int,firstStruct[fieldFirstStruct])) #estaban almacenados como strings, los paso a ints
    inMatlab = secondStruct[fieldSecondStruct]
    if isNotSameLength :
        coordsDistance = "the scanpaths don't have the same length" #si los scanpaths no tienen la misma longitud, no me importa el camino que hicieron
        differentPaths = False
    else:
        #si los scanpaths tienen la misma longitud, me fijo cuanto difieren ambos recorridos (medida en valor absoluto)
        coordsDistance = list(map(abs, [x - y for x, y in zip(inPython, inMatlab)]))
        differentPaths = bool(sum(list(map(lambda x: int(x> 100), coordsDistance))))
        #map() no devuelve una lista, por eso uso list()
    return{"python coords" :inPython, "matlab coords" : inMatlab, "coords distance" : coordsDistance, "different paths?" : differentPaths}
    

main()
