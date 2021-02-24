clear all
close all
clc

dirdata = '~/Dropbox/10_datos_pesados/vs_models_data/';

dirtodo = [dirdata 'todo/'];
dirmat  = [dirdata 'mat/'];
dirin   = [dirdata 'info/'];
dirsmp  = [dirdata 'sinfo/'];

dirout  = dirsmp;

% create_info_from_raw_data_new('../../../data/vs_models/all_data/')
% system(['cp ../../../data/vs_models/all_data/*_info.mat ' dirin])

TargetThreshold = 0;    % Margin to asign a fixation to the target
OutsideThreshold= 100;  % If a fixation falls 'OutsideThreshold' distance 
                        % outside the image, it is erased. If it falls 
                        % between 'OutsideThreshold' and the border, it is 
                        % asigned to the closed image point

tmp=dir([dirmat '*.mat']); tmp = {tmp.name};
subjnames = cellfun(@(x) x(1:end-4),tmp,'UniformOutput',0);
Nsubj = length(subjnames);

for su = 1:Nsubj
    fprintf('**********************************************************\n')
    fprintf('%s (%d / %d)\n',subjnames{su},su,length(subjnames))
    
    matfile = [subjnames{su} '.mat'];
    filein  = [subjnames{su} '_info.mat'];
    fileout = [subjnames{su} '_simple.mat'];
    load([dirmat matfile]);
    load([dirin filein]);

    smpinfo = [];
    for tr = 1:length(info)
        % IDs
        smpinfo(tr).subj            = info(tr).subj;
        smpinfo(tr).image_name      = info(tr).image;
        smpinfo(tr).target_name     = info(tr).template;

        % Stim info
        smpinfo(tr).trial_order     = info(tr).tr;
        smpinfo(tr).image_size      = info(tr).exp_data.imgsize;    % [Y X]
        smpinfo(tr).screen_size     = [exp.cfg.winh exp.cfg.winw];  % [Y X]

        mar   = (smpinfo(tr).screen_size - smpinfo(tr).image_size)/2;
        smpinfo(tr).screen_rect     = [-mar(2) -mar(1) ...
            smpinfo(tr).image_size(2)+mar(2) smpinfo(tr).image_size(1)+mar(1)];
                                                                    % [left top rigth bottom]

        smpinfo(tr).target_center   = info(tr).target_center;
        smpinfo(tr).target_rect     = [info(tr).target_center info(tr).target_center] + ...
                                        info(tr).template_side_length*[-.5 -.5 .5 .5];
                                                                    % [left top rigth bottom]
        smpinfo(tr).nsaccades_allowed = info(tr).exp_data.nsaccades_thr;
        smpinfo(tr).initial_position= info(tr).exp_data.pos - mar;  % [X Y] 

        % response info
        smpinfo(tr).target_found    = info(tr).exp_data.stopcriteria==2;
        smpinfo(tr).search_rt       = info(tr).exp_data.rt;             % secs from trial ini
        smpinfo(tr).objetive_rt     = info(tr).exp_response.time(1);    % secs from trial ini
        smpinfo(tr).subjetive_rt    = info(tr).exp_response.time(2);    % secs from trial ini
        smpinfo(tr).objetive_pos    = info(tr).exp_response.position;   % pixels %%%%%%%%%%%%%%
        smpinfo(tr).subjetive_size  = info(tr).exp_response.size;       % pixels

        % scanpath
        smpinfo(tr).old.x_sacc      = info(tr).saccades(:,3)' - mar(1); % pixels
        smpinfo(tr).old.y_sacc      = info(tr).saccades(:,4)' - mar(2); % pixels
        smpinfo(tr).old.x           = info(tr).fixations(:,1)' - mar(1); % pixels
        smpinfo(tr).old.y           = info(tr).fixations(:,2)' - mar(2); % pixels
        smpinfo(tr).old.t           = info(tr).timeFix(:,1)';           % ms
        smpinfo(tr).old.dur         = info(tr).timeFix(:,3)';           % ms

        if (smpinfo(tr).target_found==1)
            fix2target = ( smpinfo(tr).old.x >= smpinfo(tr).target_rect(1)-TargetThreshold & ...
                    smpinfo(tr).old.x <= smpinfo(tr).target_rect(3)+TargetThreshold) & ...
                ( smpinfo(tr).old.y >= smpinfo(tr).target_rect(2)-TargetThreshold & ...
                    smpinfo(tr).old.y <= smpinfo(tr).target_rect(4)+TargetThreshold );

            if sum(fix2target)>0
                smpinfo(tr).x               = smpinfo(tr).old.x(1:find(fix2target,1,'first')) ; % pixels
                smpinfo(tr).y               = smpinfo(tr).old.y(1:find(fix2target,1,'first')) ; % pixels
                smpinfo(tr).t               = smpinfo(tr).old.t(1:find(fix2target,1,'first')) ; % pixels
                smpinfo(tr).dur             = smpinfo(tr).old.dur(1:find(fix2target,1,'first')) ; % pixels
            else
                smpinfo(tr).x               = smpinfo(tr).old.x; % pixels
                smpinfo(tr).y               = smpinfo(tr).old.y; % pixels
                smpinfo(tr).t               = smpinfo(tr).old.t; % pixels
                smpinfo(tr).dur             = smpinfo(tr).old.dur; % pixels
            end
        else
            smpinfo(tr).x               = info(tr).fixations(:,1)'; % pixels
            smpinfo(tr).y               = info(tr).fixations(:,2)'; % pixels
            smpinfo(tr).t               = info(tr).timeFix(:,1)';   % ms
            smpinfo(tr).dur             = info(tr).timeFix(:,3)';   % ms
        end
        
        % Correct fixations outside the image
        indremove     = ( (smpinfo(tr).x < -OutsideThreshold | smpinfo(tr).x > smpinfo(tr).image_size(2)+OutsideThreshold ) | ...
                        (smpinfo(tr).y < -OutsideThreshold | smpinfo(tr).y > smpinfo(tr).image_size(1)+OutsideThreshold ) );
        
        if ( any(indremove) ); 
            fprintf('%d: %d far fixations removed\n',tr,sum(indremove) )
%             keyboard; 
            smpinfo(tr).x(indremove)  = []; % pixels
            smpinfo(tr).y(indremove)  = []; % pixels
            smpinfo(tr).t(indremove)  = [];   % ms
            smpinfo(tr).dur(indremove)= [];   % ms
        end
        
        indoutside = smpinfo(tr).x < 0;
        if any(indoutside)
%             keyboard; 
            fprintf('%d: (left displacement) %d fixations corrected\n',tr,sum(indoutside) )
            smpinfo(tr).x(indoutside)  = 0; % pixels
        end
        indoutside = smpinfo(tr).y < 0;
            if any(indoutside)
%                 keyboard; 
                fprintf('%d: (top displacement) %d fixations corrected\n',tr,sum(indoutside) )
                smpinfo(tr).y(indoutside)  = 0; % pixels
            end
        indoutside = smpinfo(tr).x > smpinfo(tr).image_size(2);
            if any(indoutside)
%                 keyboard; 
                fprintf('%d: (right displacement) %d fixations corrected\n',tr,sum(indoutside) )
                smpinfo(tr).x(indoutside) = smpinfo(tr).image_size(2); % pixels
            end
        indoutside = smpinfo(tr).y > smpinfo(tr).image_size(1);
            if any(indoutside)
%                 keyboard; 
                fprintf('%d: (bottom displacement) %d fixations corrected\n',tr,sum(indoutside) )
                smpinfo(tr).y(indoutside) = smpinfo(tr).image_size(1); % pixels
            end
        
        % Filter
        if ~isempty(smpinfo(tr).x)
            smpinfo(tr).initial_distance = ...
                    sqrt ( ( smpinfo(tr).x(1) - smpinfo(tr).initial_position(1) )^2 + ...
                            ( smpinfo(tr).y(1) - smpinfo(tr).initial_position(2) )^2 ); 
                        
            smpinfo(tr).final_target_distance = ...
                    sqrt ( ( smpinfo(tr).x(end) - smpinfo(tr).target_center(1) )^2 + ...
                            ( smpinfo(tr).y(end) - smpinfo(tr).target_center(2) )^2 ); 
        else
            smpinfo(tr).initial_distance = 9999;
            smpinfo(tr).final_target_distance = 9999;
        end

    end
    fprintf('Trials sin fijaciones = %d\n',sum([smpinfo.initial_distance] == 9999));
    save(fullfile(dirout,fileout),'smpinfo')
end
% clear info exp

%%
Tr_cat          = nan(Nsubj,4);
Tr_cat_corr     = nan(Nsubj,4);
Tr_cat_corr_filt1= nan(Nsubj,4);
Tr_cat_corr_filt2= nan(Nsubj,4);
Tr_cat_corr_filt= nan(Nsubj,4);
CATEG           = [2 4 8 12];
FILTRO1         = 72;
FILTRO2         = 36;
for su = 1:Nsubj
    fileout = [subjnames{su} '_simple.mat'];
    load(fullfile(dirout,fileout))
    
    categ       = [smpinfo.nsaccades_allowed];
    corr        = [smpinfo.target_found];
    filtro1     = [smpinfo.initial_distance]<=FILTRO1;
    filtro2     = [smpinfo.final_target_distance]<=FILTRO2;
    
    for c = 1:length(CATEG)
        Tr_cat(su,c)            = sum(categ==CATEG(c));
        Tr_cat_corr(su,c)       = sum(categ==CATEG(c) & corr);
        Tr_cat_corr_filt(su,c)  = sum(categ==CATEG(c) & corr & filtro1 & filtro2);
        Tr_cat_corr_filt1(su,c)  = sum(categ==CATEG(c) & corr & filtro1);
        Tr_cat_corr_filt2(su,c)  = sum(categ==CATEG(c) & corr & filtro2);
    end
end

Tr_cat_corr_filt2(Tr_cat==0)=nan;
Tr_cat_corr_filt1(Tr_cat==0)=nan;
Tr_cat_corr_filt(Tr_cat==0)=nan;
Tr_cat_corr(Tr_cat==0)=nan;
Tr_cat(Tr_cat==0)=nan;

clc
disp(100*nanmean(Tr_cat_corr./Tr_cat));
disp(100*nanmean(Tr_cat_corr_filt1./Tr_cat));
disp(100*nanmean(Tr_cat_corr_filt2./Tr_cat));
disp(100*nanmean(Tr_cat_corr_filt./Tr_cat));
disp([])
disp(nanmean(Tr_cat_corr));
disp(nanmean(Tr_cat_corr_filt1));
disp(nanmean(Tr_cat_corr_filt2));
disp(nanmean(Tr_cat_corr_filt));

        