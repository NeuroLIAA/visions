clear all
close all
clc

M       = 0;

addpath('../dynamic_models/utils/')


dirdata = '~/Dropbox/10_datos_pesados/vs_models_data/';

dirtodo = [dirdata 'todo/'];
dirmat  = [dirdata 'mat/'];
dirin   = [dirdata 'info/'];
dirsmp  = [dirdata 'sinfo/'];

% dir_images      = '/home/usuario/repos/vs_models/images';
% dir_templates   = '/home/usuario/repos/vs_models/templates';
dir_images      = '/home/juank/repo/vs_models/images';
dir_templates   = '/home/juank/repo/vs_models/templates';

% create_info_from_raw_data_new('../../../data/vs_models/all_data/')
% system(['cp ../../../data/vs_models/all_data/*_info.mat ' dirin])

%% Subject
tmp=dir([dirsmp '*.mat']); tmp = {tmp.name};
subjnames = cellfun(@(x) x(1:end-4),tmp,'UniformOutput',0);

dirout  = '../../../data/vs_models/sinfo_subj';
%eval(['!rm ' dirout '/*'])
for su = 1:length(subjnames)
    load([dirsmp '/' subjnames{su} '.mat'])
    info_per_subj = smpinfo;
    save([dirout '/info_per_subj_' num2str(su) '.mat'],'info_per_subj')
end
eval(['!chmod -R 777 ' dirout])

%% Images
tmp=dir([dirsmp '*.mat']); tmp = {tmp.name};
subjnames = cellfun(@(x) x(1:end-4),tmp,'UniformOutput',0);

tmp=dir([dir_images '/*.jpg']); imgnames = {tmp.name};
% imgnames = cellfun(@(x) x(1:end-4),tmp,'UniformOutput',0);

dirout  = '../../../data/vs_models/sinfo_img';
for n = 1:length(imgnames)
    info_per_img = [];
    for su = 1:length(subjnames)
        fileout = [subjnames{su} '.mat'];
        load([dirsmp fileout]);
        ind = find(strcmp({smpinfo.image_name},imgnames{n}));
        info_per_img = [info_per_img, smpinfo(ind)];
    end
    save([dirout '/info_per_img_' num2str(n) '.mat'],'info_per_img')
end
eval(['!chmod -R 777 ' dirout])
