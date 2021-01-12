%% Author: Mengmi Zhang
%% Kreiman Lab
%% web: http://klab.tch.harvard.edu/
%% Date: April 5, 2018

clear all;
close all;
clc;


stimuliFolder = 'stimuli/';
enumeratedImages = dir([stimuliFolder '*.jpg']);
objfileID = fopen('croppednaturaldesign_img.txt','w');

for j = 1: length(enumeratedImages)
    trialname = enumeratedImages(j).name;
    img = rgb2gray(imread([stimuliFolder trialname]));
    img = imresize(img, [1028 1280]);
    
    imgID = trialname(4:end-4);
    mkdir('choppednaturaldesign/', ['img' imgID]);
    
    fun = @(block_struct) imwrite(block_struct.data,['choppednaturaldesign/img' imgID '/img_id' imgID '_' num2str(block_struct.location(1)) '_' num2str(block_struct.location(2)) '.jpg']);    
    blockproc(img,[224 224],fun);
    
    chopdir = dir(['choppednaturaldesign/img' imgID '/img_id' imgID '_*.jpg']);
    for i = 1: length(chopdir)
        fprintf(objfileID,'%s\n',['choppednaturaldesign/img' imgID '/' chopdir(i).name]);
    end
end

fclose(objfileID);

