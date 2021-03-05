from skimage import io, transform, img_as_ubyte
from os import listdir, path, mkdir, rename

datasetDir = '../data_images/images/'
targetsDir = '../data_images/templates/'
datasetDirOld = '../data_images/images_old/'
targetsDirOld = '../data_images/templates_old/'
stimuli_size = (1028, 1280)

def main():
        
    if not(path.exists(datasetDirOld)):
        mkdir(datasetDirOld)
    if not(path.exists(targetsDir)):
        mkdir(targetsDir)
    for template_file in listdir(targetsDir):
        if not(template_file.endswith('.jpg')):
            continue
        rename(targetsDir + '/' + template_file, targetsDirOld + '/' + template_file)
    for image_file in listdir(datasetDir):
        if not(image_file.endswith('.jpg')):
            continue
        rename(datasetDir + '/' + image_file, datasetDirOld + '/' + image_file)
        image_resized = io.imread(datasetDirOld + '/' + image_file)
        image_resized = transform.resize(image_resized, stimuli_size)
        io.imsave(datasetDir + '/' + image_file, img_as_ubyte(image_resized), check_contrast=False)
        #falta el crop
        
main()
