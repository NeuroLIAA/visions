function W = templateResponse(cfg, visibility_map)
    % templateResponse: Creates W(a,b,c,d) Where W(:,:,c,d) represents how similar the position 
    % (a,b) is to the target to the observer who is fixing his view in (c,d). This model 
    % achieves inhibition of return as well as moderate saccade lengths.
    % W is defined as shown in Geisler et al 2005

    
    % Input:
    % cfg.imgname                 = Name of the image
    % cfg.size_prior              = Size of the prior
    % cfg.target_center_top_left  = top left extreme of the target
    % cfg.target_center_bot_right = bottom right extreme of the target
    % visibility_map              = visibility map
    % cfg.delta                   = Size of the grid cells
    % cfg.dinamic_model           = Dinamic model name: 'geisler' | 'correlation'
    % cfg.target_size             = Size of the targeat
    % cfg.a                       = 
    % cfg.b                       = 

    % Output:
    % W(a,b,c,d) Where W(:,:,c,d) represents how similar the position (a,b) is to the target to 
    % the observer who is fixing his view in (c,d)

    % Creates mu. Each cell of mu has value 0.5 if the target is in that position and -0.5 if it is not
    mu = ones(cfg.size_prior(1), cfg.size_prior(2), cfg.size_prior(1), cfg.size_prior(2)) * -0.5;
    for x = cfg.target_center_top_left(1):cfg.target_center_bot_right(1)
        for y = cfg.target_center_top_left(2):cfg.target_center_bot_right(2)
            mu(x, y, :, :) = ones(cfg.size_prior(1), cfg.size_prior(2)) * 0.5;
        end
    end
    
    % visibility_map is in [0, 1]
    for i = 1:cfg.size_prior(1)
        for j = 1:cfg.size_prior(2)
            visibility_map(:,:,i,j) = visibility_map(:,:,i,j) - min(min(visibility_map(:,:,i,j)));
            visibility_map(:,:,i,j) = visibility_map(:,:,i,j) / max(max(visibility_map(:,:,i,j)));
        end
    end
        
    if strcmp(cfg.dinamic_model, 'geisler')
        Mu = mu;
    elseif strcmp(cfg.dinamic_model, 'correlation') || strcmp(cfg.dinamic_model, 'greedy') 
        % Correlation between the image and the target
        img = imread(['../data_images/images/' cfg.imgname]);
        tmp = dir(['../data_images/templates/' cfg.imgname(1:end-4) '*']);
        template = imread(['../data_images/templates/' tmp.name]);

        correlation = normxcorr2(template, img);
        correlation = correlation(cfg.target_size(1)/2:size(correlation, 1)-cfg.target_size(1)/2, cfg.target_size(1)/2:size(correlation, 2)-cfg.target_size(1)/2);
        correlation = reduceMatrix(correlation, cfg.delta, 'max');

        correlation = correlation - min(correlation(:));
        correlation = correlation / max(correlation(:)) - 0.5;
        rep_corr = repmat(correlation, [1, 1, cfg.size_prior(1), cfg.size_prior(2)]);

        % geisler_truth and rep_corr go between [-0.5,0.5], while
        % visibility_map is in [0, 1]
        Mu = mu .* (visibility_map + 0.5) + rep_corr .* (1 - visibility_map + 0.5);
        Mu = Mu / 2;
    else
        fprintf('Error: Wrong mode \n')
    end
    sigma = ones(size(visibility_map)) ./ (visibility_map*cfg.a+cfg.b);
    W = sigma .* randn(cfg.size_prior(1), cfg.size_prior(2), cfg.size_prior(1), cfg.size_prior(2)) + Mu;
end

% Geisler only takes into account that visibility affects how well we judge
% W, but does not think about possible distractors (i.e., elements that
% look alike the real template). So we can take 
