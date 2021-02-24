% DEPRECATED!! GB 11-3-2020

close all
clear all
clc

addpath('utils/')
delta       = 40; % TOMAR COMO PARAMETRO!!!

% for sub = 1:57
%     sub
%     path = strcat('../matrix/subjects/info_per_subj_subj_',num2str(sub),'.mat');
%     load(path)
%     info_per_subj = subjMapFixationToMatrix(info_per_subj, path, delta);
% end

for imgnum = 1:134
    imgnum
    path = strcat('../matrix/images/info_per_subj_img_',num2str(imgnum),'.mat');
    load(path)
    info_per_subj = subjMapFixationToMatrix(info_per_subj, path, delta);
end