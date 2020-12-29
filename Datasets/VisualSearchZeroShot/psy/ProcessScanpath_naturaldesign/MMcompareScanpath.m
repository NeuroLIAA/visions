clear all; close all; clc;
load('../../SubjectArray/naturaldesign.mat');
load('../../SubjectArray/naturaldesign_seq.mat');
[B,seq] = sort(seq);
NumTrials = length(MyData);

load('../DataForPlot/machine_croppednaturaldesign_Lscale.mat');

Subjlist = {'subj02-az','subj03-el','subj04-ni','subj05-mi'};

indworse = [];
for subj = 1: length(Subjlist)
    
    load([Subjlist{subj} '.mat']);
    TargetFound = FixData.TargetFound(:,2:end);
    TargetFound = TargetFound(seq,:);
    TargetFound = TargetFound(1: length(seq)/2,:);

    for trial = 1: length(seq)/2
        human = TargetFound(trial,:);
        machine = scoremat(trial,1:79);
        stephuman = find(human == 1);
        stepmachine = find(machine == 1);
        
        if isempty(stephuman)
            continue;
        else
            if isempty(stepmachine)
                indworse = [indworse; trial];
            else
                if stephuman < stepmachine
                    indworse = [indworse; trial];
                end
            end
        end
    end
end

U = unique(indworse);
indworseall =  U(   find (hist(indworse,U) >=3 ) );

screeny = 1024;
screenx = 1280;

examplesubj = 2;
load([Subjlist{examplesubj} '.mat']);
Fix_posx = FixData.Fix_posx(seq);
Fix_posy = FixData.Fix_posy(seq);

load('../DataForPlot/Fix_croppednaturaldesign.mat');

for i = 1: length(indworseall)
    badind = indworseall(i);
    
    %to check fixations; plot them out
    img = imread(['../../../Datasets/NaturalDesign/img' sprintf('%03d',badind) '.jpg']);
    img = rgb2gray(img);
    %img = imread(['/home/mengmi/Proj/Proj_VS/Datasets/NaturalDataset/filtered/img' sprintf('%03d',badind) '.jpg']);
    img = imresize(img,[screeny screenx]);

     fixx = Fix_posx{badind};
     fixy = Fix_posy{badind};
     
     
     subplot(2,2,1);
     fixnumstr = cellstr(string([1:1:length(fixx)]));
     RGB = insertText(img,[int32(fixx); int32(fixy)]', fixnumstr,'FontSize',28);
     imshow(RGB);
     title(['human; img# ' num2str(badind)]);
     
     
     fixx = Fix.FixX{badind}';
     fixy = Fix.FixY{badind}';
     
     subplot(2,2,2);
     fixnumstr = cellstr(string([1:1:length(fixx)]));
     RGB = insertText(img,[int32(fixy); int32(fixx)]', fixnumstr,'FontSize',28);
     imshow(RGB);
     title('machine');
     
     subplot(2,2,3);
     gt = imread(['../../../Datasets/NaturalDesign/gt' num2str(badind) '.jpg']);
     imshow(gt);
     
     subplot(2,2,4);
     target = imread(['../../../Datasets/NaturalDesign/t' sprintf('%03d',badind) '.jpg']);
     target = rgb2gray(target);
     imshow(target);
     pause;
     
end
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     