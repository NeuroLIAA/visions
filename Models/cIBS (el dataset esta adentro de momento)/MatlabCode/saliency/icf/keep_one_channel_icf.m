% folderPath1 = Folder with 3-channel images
folderPath1 = '/home/gastonb/Repos/vs-aux/vs_models-master/saliency/icf/aux';
cd(folderPath1); % path of the folder
% WriteDir = Folder to save 1-channel images
WriteDir = '/home/gastonb/Repos/vs-aux/vs_models-master/saliency/icf';
files1 = dir('**');
files1(1:2) = [];
totalFiles = numel(files1);
for i =1:totalFiles
    Fileaddress{i,1}=strcat(folderPath1,'/',files1(i).name);
    file{i} = imread(Fileaddress{i,1});
    % Edit the file
    img = file{i}(:,:,1);
    cd(WriteDir) % go to dir where you want to save updated files
    %writeFileName = strcat(.jpg');
    imwrite(img, files1(i).name)
    cd(folderPath1) % return to actualFile folder
end
