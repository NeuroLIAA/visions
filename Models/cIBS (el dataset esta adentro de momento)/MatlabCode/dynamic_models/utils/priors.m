function prior = priors(cfg)
    % PRIORS: Load prior probability of target being in each location
    % p = priors(cfg) creates a matrix of prior probability of target being 
    % in each location with mode cfg.static_model
    % Dimensions are determined by floor(size(saliencyMap) / cfg.delta)
    
    % Input:
    % cfg.imgname       = Name of the image  
    % cfg.delta         = Size of grid cells
    % cfg.static_model  = Static model name: 'mlnet' | 'judd1' | 'judd2' | 'judd3' | 'judd4' | 'judd5' | 'flat' | 'center' | 'noisy'
    % cfg.image_size    = Size of the image
    
    switch cfg.static_model
        case 'deepgaze'
            saliencyMap = imread(['../saliency/deepgaze/' cfg.imgname]);
        case 'icf'
            saliencyMap = imread(['../saliency/icf/' cfg.imgname]);
        case 'sam-resnet'
            saliencyMap = imread(['../saliency/sam_resnet/' cfg.imgname]);
        case 'sam-vgg'
            saliencyMap = imread(['../saliency/sam_vgg/' cfg.imgname]);
        case 'mlnet'
            saliencyMap = imread(['../saliency/mlnet/' cfg.imgname]);
        case 'judd1'
            saliencyMap = imread(['../saliency/judd_model_1/all_subjects_fix_3_' cfg.imgname]);
        case 'judd2'
            saliencyMap = imread(['../saliency/judd_model_2/all_subjects_fix_3_' cfg.imgname]);
        case 'judd3'
            saliencyMap = imread(['../saliency/judd_model_3/all_subjects_fix_3_' cfg.imgname]);
        case 'judd4'
            saliencyMap = imread(['../saliency/judd_model_4/all_subjects_fix_3_' cfg.imgname]);
        case 'judd5'
            saliencyMap = imread(['../saliency/judd_model_5/all_subjects_fix_3_' cfg.imgname]);
        case 'flat'
            saliencyMap = ones(cfg.image_size);
        case 'center'
            % from findDistToCenterFeatures(img, dims), Judd et al ICCV 2009
            % run findDistToCenterFeatures(ones(768, 1024), [768, 1024], 768/2, 1024/2)
            saliencyMap = imread(['../saliency/center.jpg']);
        case 'noisy'
            saliencyMap = awgn(ones(cfg.image_size), 25); % White Gaussian Noise
        otherwise
            fprintf('Error: Wrong mode \n')
    end
    prior = reduceMatrix(saliencyMap, cfg.delta, 'mean');
end
