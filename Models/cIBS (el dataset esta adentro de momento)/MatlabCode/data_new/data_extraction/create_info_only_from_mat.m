function create_info_only_from_mat(filename, dir_data)
    % filename = for example, 'MS0101'
    % dir_data = directory where 'MS0101.mat' can be found
    if nargin < 2
        dir_data = 'experiment_data/';
    end
    addpath('utils');
    read_evt(filename, dir_data, false);
    expand_info(dir_data);
end

function read_evt(filename, dir_data, will_plot)
    % READ_EVT: Retrieve eye data from EVT field in .mat instead of EDF
    % This way of getting data can be noisier than EDF. In particular, it was
    % detected that for many images there is one more fixation in one file than
    % in the other due to small differences in recording time. In rare cases,
    % there are 2 fixations more in one recording than the other. When a
    % fixation was recorded in both files, it appears to be precise in both.

    % will_plot = true if we want to plot all scanpaths (one per figure), false otherwise
    load([dir_data filename '.mat'])
    el = exp.el;

    load('../matrices/target_positions_final.mat');
    sorted_images = {target_positions.image};
    fieldNames = fieldnames(target_positions(1));

    EY = [];

    info = [];
    todo = struct('refix', [], 'resac', []);
    for tr = 1:length(exp.data)
        N = length(exp.data(tr).evt); 
        evt = [];
        for i=1:N
            if ~isempty(exp.data(tr).evt(i).evt)
                evt = [evt exp.data(tr).evt(i).evt];
            end
        end
        N = length(evt);

        % {STARTBLINK: 3, ENDBLINK: 4, STARTSACC: 5, ENDSACC: 6, STARTFIX: 7, ENDFIX: 8}
        % type: ideal: [7 8 5 6 7 8 ...] o [5 6 7 8 5 6...] pero es imposible y
        % siempre va a empezar con una sacada o una fijacion cortada. Entonces,
        ty = [evt.type];

        % saco los blink events para el analisis de bordes
        ty = ty(ismember(ty,[5,6,7,8]));

        t0 = evt(1).time;
        sactini = [evt([evt.type]==5).time]' - t0;
        sactend = [evt([evt.type]==6).time]' - t0;
        fixtini = [evt([evt.type]==7).time]' - t0; 
        fixtend = [evt([evt.type]==8).time]' - t0;
        if (ty(1)==8);      fixtini = [fixtini(1)-1000; fixtini];   end
        if (ty(end)==7);    fixtend = [fixtend; fixtend(end)+1000]; end

        if (ty(1)==6);      sactini = [sactini(1)-1000; sactini];   end
        if (ty(end)==5);    sactend = [sactend; sactend(end)+1000]; end

        if length(sactini) ~= length(sactend) || length(fixtini) ~= length(fixtend)
            continue;
        end
        sact = [sactini sactend];
        fixt = [fixtini fixtend];

        % En general empiezan y terminan con una fijacion.
        tmp = struct();
        tmp.Nfix = size(fixt,1);
        tmp.Nsac = size(sact,1);

        % Para la posicion.
        % gstx, gsty: gaze start x, gaze start x: esta bien definido para 5 y 7
        % genx, geny: gaze end x, gaze end x: esta bien definido para 6 y 8
        % gavx, gavy: gaze average x, gaze average x: esta bien definido para 6 y 8
        sactini = [evt([evt.type]==5).time]';
        sactend = [evt([evt.type]==6).time]';
        fixtini = [evt([evt.type]==7).time]'; 
        fixtend = [evt([evt.type]==8).time]';

        sacgstx = [evt([evt.type]==5).gstx]';
        sacgenx = [evt([evt.type]==6).genx]';
        fixgstx = [evt([evt.type]==7).gstx]'; 
        fixgenx = [evt([evt.type]==8).genx]';
        fixgavx = [evt([evt.type]==8).gavx]';

        sacgsty = [evt([evt.type]==5).gsty]';
        sacgeny = [evt([evt.type]==6).geny]';
        fixgsty = [evt([evt.type]==7).gsty]'; 
        fixgeny = [evt([evt.type]==8).geny]';
        fixgavy = [evt([evt.type]==8).gavy]';

        fixaps  = [evt([evt.type]==8).ava]'; % average pupil size
        sacpvel  = [evt([evt.type]==6).pvel]'; % peak velocity

        if (ty(1)==8);      
            fixtini = [NaN; fixtini];   
            fixgstx = [NaN; fixgstx];   
            fixgsty = [NaN; fixgsty];   
        end
        if (ty(end)==7);    
            fixtend = [fixtend; NaN]; 
            fixgenx = [fixgenx; NaN]; 
            fixgeny = [fixgeny; NaN]; 
            fixgavx = [fixgavx; NaN]; 
            fixgavy = [fixgavy; NaN];
            fixaps  = [fixaps; NaN];
        end

        if (ty(1)==6);      
            sactini = [NaN; sactini];   
            sacgstx = [NaN; sacgstx];   
            sacgsty = [NaN; sacgsty];
        end
        if (ty(end)==5);    
            sactend = [sactend; NaN]; 
            sacgenx = [sacgenx; NaN]; 
            sacgeny = [sacgeny; NaN];
        end

        % tini tend dur gavx gavy aps gstx gsty genx geny
        tmp.fix = [fixtini,fixtend,fixtend-fixtini,fixgavx,fixgavy,fixaps,fixgstx,fixgsty,fixgenx,fixgeny,...
            nanmean([fixgstx,fixgenx,fixgavx],2),nanmean([fixgsty,fixgeny,fixgavy],2)];

        % tini tend dur gstx gsty genx geny
        tmp.sac = [sactini,sactend,sactend-sactini,sacgstx,sacgsty,sacgenx,sacgeny];

        % add fields to "todo"
        todo.refix = [todo.refix; tmp.fix(:, 1:6)];
        todo.resac = [todo.resac; [tmp.fix(:, 1:7), nan(size(tmp.fix(:, 1:7), 1), 1), nan(size(tmp.fix(:, 1:7), 1), 1)]];

        img_info = target_positions(find(cellfun(@(x) ~isempty(strfind(x, exp.trial(tr).imgname)), sorted_images)));
        fixations = tmp.fix(:,end-1:end) - 128;
	timeFix = tmp.fix(:,1:3);
        %save(sprintf('fix%d_evt.mat', tr), 'fixations');

        t.fixations = fixations;
	t.timeFix = timeFix;
        for j = 1:length(fieldNames)
            t.(fieldNames{j}) = img_info.(fieldNames{j});
        end
        t.tr = tr;
        info = [info t];

        if will_plot
            plot_scanpath(info, exp, tr, filename(1:end-4))
        end

        EY = [EY tmp];
    end

    save([dir_data filename '_info.mat'], 'info', 'exp');
end
