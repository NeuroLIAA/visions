from skimage import io, measure, color, transform
from os import listdir
import json
import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def getName(imgID, _type):
    strNumber = str(imgID)
    if imgID < 100:
        strNumber = '0' + strNumber
    if imgID < 10:
        strNumber = '0' + strNumber

    if (_type == 'image'):
        name = 'img' + strNumber + '.jpg'
    else:
        name = 't' + strNumber + '.jpg'

    return name

gtDir = '../gt/'
gtFiles = sorted_alphanumeric(listdir(gtDir))

target_positions = []
for gt in gtFiles:
    imgID = gt[2:-4]
    gtImg = io.imread(gtDir + gt)
    mask = color.rgb2gray(gtImg) > 0.5
    # Label target region
    label_gtImg = measure.label(mask)
    # Get target region
    target = measure.regionprops(label_gtImg)
    start_row, start_column, end_row, end_column = target[0].bbox
    target_height = end_row - start_row
    target_width = end_column - start_column

    img_height = gtImg.shape[0]
    img_width  = gtImg.shape[1]

    imgName = getName(int(imgID), 'image')
    tgName  = getName(int(imgID), 'target')

    target_positions.append({ "image" : imgName, "target" : tgName, "dataset" : "IVSN Natural Design Dataset", "target_matched_row" : start_row, "target_matched_column" : start_column, \
         "target_height" : target_height, "target_width" : target_width, "image_height" : img_height, "image_width" : img_width, "initial_fixation_row" : 511, "initial_fixation_column" : 639})

jsonStructsFile = open('../trials_properties.json', 'w')
json.dump(target_positions, jsonStructsFile, indent = 4)
jsonStructsFile.close()
