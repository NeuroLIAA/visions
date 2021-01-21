stimuliFolder = 'stimuli/';
enumeratedImages = dir([stimuliFolder '*.jpg']);

for j = 1: length(enumeratedImages)
    trialname = enumeratedImages(j).name;
    img = imread([stimuliFolder trialname]);
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    imwrite(img, [stimuliFolder trialname])
end