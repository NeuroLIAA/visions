from skimage import io, measure, color, transform
import json
from os import listdir

def getName(imgID, _type):
    strNumber = str(imgID)
    if imgID < 100:
        strNumber = '0' + strNumber
    if imgID < 10:
        strNumber = '0' + strNumber

    if (_type == 'image'):
        name = 'img' + strNumber + '.jpg'
    else:
        name = 'tg' + strNumber + '.jpg'

    return name

gtDir = 'stimuli/gt/'
gtFiles = listdir(gtDir)

imgID = 1
target_positions = []
for gt in gtFiles:
    gtImg = io.imread(gtDir + gt)
    mask = color.rgb2gray(gtImg) > 0.5
    # Label target region
    label_gtImg = measure.label(mask)
    # Get target region
    target = measure.regionprops(label_gtImg)
    start_row, start_column, end_row, end_column = target[0].bbox
    target_side_length = end_row - start_row
    target_columns = end_column - start_column

    img_height = gtImg.shape[0]
    img_width  = gtImg.shape[1]

    imgName = getName(imgID, 'image')
    tgName  = getName(imgID, 'target')

    target_positions.append({ "image" : imgName, "template" : tgName, "matched_row" : start_row, "matched_column" : start_column, "target_side_length" : target_side_length, \
         "target_columns" : target_columns, "image_height" : img_height, "image_width" : img_width})
    imgID += 1

jsonStructsFile = open('target_positions.json', 'w')
json.dump(target_positions, jsonStructsFile, indent = 4)
jsonStructsFile.close()