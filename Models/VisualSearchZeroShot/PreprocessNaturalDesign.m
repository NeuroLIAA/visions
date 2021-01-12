%% Author: Mengmi Zhang
%% Kreiman Lab
%% web: http://klab.tch.harvard.edu/
%% Date: April 5, 2018

clear all;
close all;
clc;

stimuliFolder = 'stimuli/';
enumeratedImages = dir([stimuliFolder '*.jpg']);

for j = 1: length(enumeratedImages)
    trialname = enumeratedImages(j).name;
    img = imread([stimuliFolder trialname]);
    if ndims(img) == 3
        img = rgb2gray(img);
    end

    img = imresize(img, [1028 1280]);
    
    imgID = trialname(4:end-4);
    
    if ~exist(['choppednaturaldesign/img' imgID], 'dir')
       mkdir(['choppednaturaldesign/img' imgID]);
    end

    
    fun = @(block_struct) imwrite(block_struct.data,['choppednaturaldesign/img' imgID '/img_id' imgID '_' num2str(block_struct.location(1)) '_' num2str(block_struct.location(2)) '.jpg']);    
    blockproc(img,[224 224],fun);
end

