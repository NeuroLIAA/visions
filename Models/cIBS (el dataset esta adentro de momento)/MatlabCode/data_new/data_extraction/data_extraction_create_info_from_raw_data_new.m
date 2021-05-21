function create_info_from_raw_data_new(dir_data)
    % CREATE_INFO_FROM_RAW_DATA Take X.asc and X.mat and create X_info.mat
    % with all the combined information

    % dir_data = directory where 'MS0101.mat' and 'MS0101.asc' can be found

    % (09/03/2020, JK) Same as create_info_from_raw_data() withoout the
    % displacement of 128 pixels on the fixation positions

    if nargin == 0
        dir_data = './experiment_data/';
    end
    addpath('utils');
    
    create_todo_from_asc(dir_data);
    create_info_from_todo(dir_data);
    expand_info(dir_data);
end

function create_todo_from_asc (dir_data)
    tmp = dir([dir_data '*.asc']); filenames = {tmp.name};

    data = [];
    for su = 1:length(filenames)
        if ~exist([dir_data upper(filenames{su}(1:end-4)) '_todo.mat'], 'file')
            [todo] = fun_extract_all_eye_data(filenames{su}, dir_data);
            save([dir_data upper(filenames{su}(1:end-4)) '_todo.mat'], 'todo')
        end
    end
end

function create_info_from_todo (dir_data)
    try
        tmp = dir([dir_data '*_todo.mat']); filenames = {tmp.name};

        load('../matrix/target_positions_final.mat');
        fieldNames = fieldnames(target_positions(1));
        sorted_images = {target_positions.image};

        for su = 1:length(filenames)
            load([dir_data filenames{su}]);
            load([dir_data upper(filenames{su}(1:end-9)) '.mat']);
            filenames{su}
            msg = [todo.msg];
            msgtime = [todo.msgtime];

            ind_bgn = find(cellfun(@(x) ~isempty(strfind(x,'startsearch')),[todo.msg])); % tambien puede ser fixcross
            ind_end = find(cellfun(@(x) ~isempty(strfind(x,'endsearch')),[todo.msg]));

            t_bgn = msgtime(ind_bgn);
            t_end = msgtime(ind_end);

            Nfix = nan(length(exp.response),1);
            Nsac = nan(length(exp.response),1);
            tr_inicial = find(cellfun(@(x) ~isempty(x), {exp.response.size}), 1)-1;

            info = [];
            for tr = (tr_inicial+1):length(exp.response)
                % fixations: '<stime> <etime> <dur> <axp> <ayp> <aps>'
                indfix = (todo.refix(:,2)>t_bgn(tr-tr_inicial) & todo.refix(:,1)<t_end(tr-tr_inicial));
                Nfix(tr) = sum(indfix);
                if Nfix(tr)>0
                    fixations = nan(Nfix(tr), 2);
                    timeFix = nan(Nfix(tr), 3);

                    valid_fix = find(indfix==1);
                    for i = 1:length(valid_fix)
                        fixations(i,:)=[todo.refix(valid_fix(i),4) todo.refix(valid_fix(i),5)];
                        timeFix(i,:)=[todo.refix(valid_fix(i),1) todo.refix(valid_fix(i),2) todo.refix(valid_fix(i),2)-todo.refix(valid_fix(i),1)];
                    end
                    fixations = fixations;

                    img_info    = target_positions(find(cellfun(@(x) ~isempty(strfind(x, exp.trial(tr).imgname)), sorted_images)));
                    t.fixations = fixations;
                    t.timeFix = timeFix;
                    for j = 1:length(fieldNames)
                        t.(fieldNames{j}) = img_info.(fieldNames{j});
                    end
                else
                    indfix = (todo.lefix(:,2)>t_bgn(tr-tr_inicial) & todo.lefix(:,1)<t_end(tr-tr_inicial));
                    Nfix(tr) = sum(indfix);

                    fixations = nan(Nfix(tr), 2);
                    timeFix = nan(Nfix(tr), 3);

                    valid_fix = find(indfix==1);
                    for i = 1:length(valid_fix)
                        fixations(i,:)=[todo.lefix(valid_fix(i),4) todo.lefix(valid_fix(i),5)];
                        timeFix(i,:)=[todo.lefix(valid_fix(i),1) todo.lefix(valid_fix(i),2) todo.lefix(valid_fix(i),2)-todo.lefix(valid_fix(i),1)];
                    end

                    img_info    = target_positions(find(cellfun(@(x) ~isempty(strfind(x, exp.trial(tr).imgname)), sorted_images)));
                    t.fixations = fixations;
                    t.timeFix = timeFix;
                    for j = 1:length(fieldNames)
                        t.(fieldNames{j}) = img_info.(fieldNames{j});
                    end
                end

                % Saccades (2020-03-11, JK)
                % saccades: '<stime> <etime> <dur> <sxp> <syp> <exp> <eyp> <ampl> <pv>'
                indsac = (todo.resac(:,2)>t_bgn(tr-tr_inicial) & todo.resac(:,1)<t_end(tr-tr_inicial));
                Nsac(tr) = sum(indsac);
                if Nsac(tr)>0
                    saccades    = nan(Nsac(tr), 4);
                    timeSacc    = nan(Nsac(tr), 3);

                    valid_sac = find(indsac==1);
                    for i = 1:length(valid_sac)
                        saccades(i,:)=[todo.resac(valid_sac(i),4) todo.resac(valid_sac(i),5) todo.resac(valid_sac(i),6) todo.resac(valid_sac(i),7)];
                        timeSacc(i,:)=[todo.resac(valid_sac(i),1) todo.resac(valid_sac(i),2) todo.resac(valid_sac(i),2)-todo.resac(valid_sac(i),1)];
                    end

                    t.saccades = saccades;
                    t.timeSacc = timeSacc;
                else
                    indsac = (todo.lesac(:,2)>t_bgn(tr-tr_inicial) & todo.lesac(:,1)<t_end(tr-tr_inicial));
                    Nsac(tr) = sum(indsac);

                    saccades    = nan(Nsac(tr), 4);
                    timeSacc    = nan(Nsac(tr), 3);

                    valid_sac = find(indsac==1);
                    for i = 1:length(valid_sac)
                        saccades(i,:)=[todo.lesac(valid_sac(i),4) todo.lesac(valid_sac(i),5) todo.lesac(valid_sac(i),6) todo.lesac(valid_sac(i),7)];
                        timeSacc(i,:)=[todo.lesac(valid_sac(i),1) todo.lesac(valid_sac(i),2) todo.lesac(valid_sac(i),2)-todo.lesac(valid_sac(i),1)];
                    end

                    t.saccades = saccades;
                    t.timeSacc = timeSacc;
                end
                t.tr = tr;
                info = [info t];
            end
            save([dir_data upper(filenames{su}(1:end-9)) '_info.mat'], 'info', 'exp');
        end
    catch ME
        disp('Ouch!')
        keyboard;
    end
end
