    function bayesianSaliencyModel(cfg)
    % BAYESIANSALIENCYMODEL: visual search model based on Najemnik and Geisler 2005
    
    % Input:
    % cfg.static_model    = Static model name: 'mlnet' | 'judd1' | 'judd2' | 'judd3' | 'judd4' | 'judd5' | 'flat' | 'center' | 'noisy'
    % cfg.dinamic_model   = Dinamic model name: 'greedy' | 'geisler' | 'correlation'
    % cfg.img_quantity    = Maximum image quantity
    % cfg.nsaccades_thr   = Maximum number of saccades allowed by the model
    % cfg.delta           = Size of the grid cells, used value is 32
    % cfg.target_size     = Size of the target
    % cfg.image_size      = Size of the image
    % cfg.a               = Scale factor, used value is 3
    % cfg.b               = Additive shifth, used value is 4
    % cfg.out_models_path = path where the results of the models are saved
    % cfg.initial_fix     = Initial fixation of the image expressed in pixels
    % cfg.imgnum          = Number of the image
    % cfg.imgname         = Name of the image  
    % cfg.target_center   = Position of the center of the target expressed in pixels
    
    % initialize cfg.prior (initial probability for each position)
    cfg.prior = priors(cfg);
    cfg.prior = cfg.prior / max(cfg.prior(:));

    % uniform sum of all priors (sum(cfg.prior(:)) = sumAllProbs)
    cfg.size_prior = size(cfg.prior);
    sumAllProbs = cfg.size_prior(1) * cfg.size_prior(2) * 10;
    cfg.prior = cfg.prior * (sumAllProbs - cfg.size_prior(1)*cfg.size_prior(2)) / sum(cfg.prior(:)) + 1;

    % find target borders
    cfg.target_center_top_left = mapToReducedMatrix(cfg.target_center - cfg.target_size/2, cfg.delta, cfg.image_size);
    cfg.target_center_bot_right = mapToReducedMatrix(cfg.target_center + cfg.target_size/2, cfg.delta, cfg.image_size);
    % check if the target is in the grid
    if cfg.target_center_top_left(1) <= 0 || cfg.target_center_top_left(2) <= 0 || ...
            cfg.target_center_bot_right(1) > cfg.size_prior(1) || cfg.target_center_bot_right(2) > cfg.size_prior(2)
        return;
    end
    
    % initialize k (k = fixation location at each time)
    k = nan(cfg.nsaccades_thr+1, 2);
    k(1,:) = mapToReducedMatrix(cfg.initial_fix, cfg.delta, cfg.image_size);
    % check if the first fixation is in the grid
    if k(1,1) > cfg.size_prior(1) || k(1,2) > cfg.size_prior(2) || k(1,1) <= 0 || k(1,2) <= 0
        return; 
    end
        
    % load or create visibility map
    if exist(['visibility_map_x_' num2str(cfg.size_prior(1)) '_y_' num2str(cfg.size_prior(2)) '_gaussian.mat'])
       load(['visibility_map_x_' num2str(cfg.size_prior(1)) '_y_' num2str(cfg.size_prior(2)) '_gaussian.mat']);
    else
        visibility_map = visibilityMapSimplified(cfg, 'gaussian');
        visibility_map = visibility_map - min(visibility_map(:));
        visibility_map = visibility_map / max(visibility_map(:)) * 3;
        max_x = cfg.size_prior(1);
        max_y = cfg.size_prior(2);
        save(['visibility_map_x_' num2str(cfg.size_prior(1)) '_y_' num2str(cfg.size_prior(2)) '_gaussian.mat'], 'visibility_map', 'max_x', 'max_y');
    end
    
    % initialize variables
    s = nan(cfg.size_prior(1), cfg.size_prior(2), cfg.nsaccades_thr);
    f = nan(cfg.size_prior(1), cfg.size_prior(2), cfg.nsaccades_thr);
    p = nan(cfg.size_prior(1), cfg.size_prior(2), cfg.nsaccades_thr);
    tmp = zeros(cfg.size_prior(1), cfg.size_prior(2));
    accum = zeros(cfg.size_prior(1), cfg.size_prior(2), cfg.nsaccades_thr);
    W = nan(cfg.size_prior(1), cfg.size_prior(2), cfg.size_prior(1), cfg.size_prior(2), cfg.nsaccades_thr);
    
    fprintf('   Saccade: ');
    for T = 1:cfg.nsaccades_thr
        fprintf('%d ', T);
        % initialize W(a,b,c,d) Where W(:,:,c,d) represents how similar the position (a,b) is 
        % to the target to the observer who is fixing his view in (c,d)
        W(:,:,:,:,T) = templateResponse(cfg, visibility_map);        
       
        % compute p
        if T == 1
            s(:,:,T) = W(:,:,k(T,1),k(T,2),T) .* (visibility_map(:,:,k(T,1),k(T,2)) .^ 2);
            f(:,:,T) = cfg.prior .* exp(s(:,:,T));
        else
            s(:,:,T) = s(:,:,T-1) + W(:,:,k(T,1),k(T,2),T) .* (visibility_map(:,:,k(T,1),k(T,2)) .^ 2);
            f(:,:,T) = sumAllProbs * p(:,:,T-1) .* exp(s(:,:,T));
        end
        f_all_locations = sum(sum(f(:,:,T)));
        p(:,:,T) = f(:,:,T) ./ f_all_locations;
        
        % save the probability map
        probability_map = p(:,:,T);
	if ~exist([cfg.out_models_path, '/probability_map'],'dir')
		mkdir([cfg.out_models_path, '/probability_map'])
	end
        save([cfg.out_models_path, '/probability_map/probabilityMap_Image_' num2str(cfg.imgnum) '_Saccade_' num2str(T) '.mat'], 'probability_map');
        
        % compute the next fix
        [idx_x, idx_y] = nextFix(cfg, T, p, visibility_map);
        k(T+1, :) = [idx_x idx_y];
        
        % finish search if target was found
        if cfg.target_center_top_left(1) <= k(T+1,1) && k(T+1,1) <= cfg.target_center_bot_right(1) && ...
           cfg.target_center_top_left(2) <= k(T+1,2) && k(T+1,2) <= cfg.target_center_bot_right(2)
            sprintf('target found!\n');
            break;
        end
    end
    
    % save the data
    % filter nan rows from k
    [r, ~] = find(isnan(k), 1);
    if isempty(r)
        scanpath = k;
    else
        scanpath = k(1:r-1, :);
    end
    
   
    	if ~exist([cfg.out_models_path, '/scanpath'],'dir')
		mkdir([cfg.out_models_path, '/scanpath'])
	end
	save([cfg.out_models_path, '/scanpath/scanpath_' num2str(cfg.imgnum) '.mat'], 'scanpath');
       
    	if ~exist([cfg.out_models_path, '/cfg'],'dir')
		mkdir([cfg.out_models_path, '/cfg'])
	end
	save([cfg.out_models_path, '/cfg/cfg_' num2str(cfg.imgnum) '.mat'], 'cfg');
   %keyboard
end
