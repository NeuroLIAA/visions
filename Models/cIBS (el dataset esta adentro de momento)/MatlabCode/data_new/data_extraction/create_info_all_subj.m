function create_info_all_subj(src_path, out_path)
    % GASTON - Function to create a struct with all info from sinfo_subj.
    % This struct is named info_all_subjects.
    if nargin == 0
        src_path='../new_matrix/sinfo_subj/';
        out_path='../new_matrix/';
    end
    % if info_all_subj doesnt exist, it creates
    if ~isfile(strcat(out_path,'info_all_subj.mat'))
        info_per_subj_final = [];
        for s = 1:57
            info_per_subj_aux = load(strcat(src_path,'info_per_subj_',int2str(s),'.mat'));
            [~,ind] = size(info_per_subj_aux.info_per_subj);
            for t = 1:ind
                info_per_subj_final = [info_per_subj_final info_per_subj_aux.info_per_subj(t)]; 
            end
        end

        save(strcat(out_path,'info_all_subj.mat'), 'info_per_subj_final');
    else
        fprintf('WARNING: File already exist in out_path\n')
    end
end