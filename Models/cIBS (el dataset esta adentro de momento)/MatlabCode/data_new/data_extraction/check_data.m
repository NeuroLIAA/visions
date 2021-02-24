clear all
close all
clc

dir_data = '/media/backup/Experimentos/vs_models/all_data/';

tmp = dir([dir_data '*']);          file_names  = {tmp(3:end).name};N_files = length(file_names); 
tmp = dir([dir_data '*.asc']);      asc_names   = {tmp.name};       N_asc   = length(asc_names);
tmp = dir([dir_data '*_todo.mat']); todo_names  = {tmp.name};       N_todo  = length(todo_names);
tmp = dir([dir_data '*_info.mat']); info_names  = {tmp.name};       N_info  = length(info_names);
tmp = dir([dir_data '*.mat']);      mat_names   = {tmp.name};
mat_names = mat_names(~ismember(mat_names,[todo_names info_names])); N_mat  = length(mat_names);

disp([N_files N_asc+N_todo+N_info+N_mat N_asc N_todo N_info N_mat])

extra_names = file_names(~ismember(file_names,[asc_names todo_names info_names mat_names]));
% There's some .edf files in the folder. I'm going to create a new folder
% (edffiles) and move them there.

% Now, the total number of files must be equal to de sum of the four file
% types, and all the file types should have the same number of files.
disp([N_files N_asc+N_todo+N_info+N_mat N_asc N_todo N_info N_mat])

tmp1 = upper(asc_names); tmp2 = upper(todo_names);
for i=1:length(tmp1)
    tmp1{i} = tmp1{i}(1:end-4); tmp2{i} = tmp2{i}(1:end-4);
    ind_ = strfind(tmp1{i},'_'); if ~isempty(ind_); tmp1{i} = tmp1{i}(1:ind_-1); end
    ind_ = strfind(tmp2{i},'_'); if ~isempty(ind_); tmp2{i} = tmp2{i}(1:ind_-1); end
end
tmp1 = sort(tmp1); tmp2 = sort(tmp2); u = nan(size(tmp1));
for i=1:length(tmp1); u(i) = strcmp(tmp1{i},tmp2{i}); end
if any(u==0); disp('WARNING!!!'); end

tmp1 = upper(asc_names); tmp2 = upper(info_names);
for i=1:length(tmp1)
    tmp1{i} = tmp1{i}(1:end-4); tmp2{i} = tmp2{i}(1:end-4);
    ind_ = strfind(tmp1{i},'_'); if ~isempty(ind_); tmp1{i} = tmp1{i}(1:ind_-1); end
    ind_ = strfind(tmp2{i},'_'); if ~isempty(ind_); tmp2{i} = tmp2{i}(1:ind_-1); end
end
tmp1 = sort(tmp1); tmp2 = sort(tmp2); u = nan(size(tmp1));
for i=1:length(tmp1); u(i) = strcmp(tmp1{i},tmp2{i}); end
if any(u==0); disp('WARNING!!!'); end

tmp1 = upper(asc_names); tmp2 = upper(mat_names);
for i=1:length(tmp1)
    tmp1{i} = tmp1{i}(1:end-4); tmp2{i} = tmp2{i}(1:end-4);
    ind_ = strfind(tmp1{i},'_'); if ~isempty(ind_); tmp1{i} = tmp1{i}(1:ind_-1); end
    ind_ = strfind(tmp2{i},'_'); if ~isempty(ind_); tmp2{i} = tmp2{i}(1:ind_-1); end
end
tmp1 = sort(tmp1); tmp2 = sort(tmp2); u = nan(size(tmp1));
for i=1:length(tmp1); u(i) = strcmp(tmp1{i},tmp2{i}); end
if any(u==0); disp('WARNING!!!'); end

% We checked that we have 4 files per subject.
