function visibility_map = visibilityMapSimplified(cfg, mode)
    % visibility_map: Generate a map of visibility which size is
    % cfg.size_prior. This visibility map is based on Najemnik & Geisler 2005
    
    % Input:
    % cfg.size_prior = Size of the prior 
    % cfg.delta      = Size of grid cells
    % cfg.image_size = Size of the image
    % mode           = 'gaussian' | 'flat-ellipse'
    
    %Output:
    %visibility_map(a,b,c,d) Where visibility_map(:,:,c,d) is the visibility map fixing the view in position (c,d) 

    if nargin == 1
        mode = 'flat-ellipse';
    end
    
    visibility_map = nan(cfg.size_prior(1), cfg.size_prior(2),cfg.size_prior(1),cfg.size_prior(2));   
    
    grid_size = floor(cfg.image_size / cfg.delta);
    offset = floor((cfg.image_size - grid_size*cfg.delta) / 2);
        
    if strcmp(mode, 'gaussian')
        Sigma = [4000 0; 0 2600];
        x1 = linspace(0, cfg.image_size(2), cfg.size_prior(2)); x2 = linspace(0, cfg.image_size(1), cfg.size_prior(1));

        [X1,X2] = meshgrid(x1,x2);

        for ix = 1:cfg.size_prior(1)
            for iy = 1:cfg.size_prior(2)
                fixation = [ix iy] * cfg.delta - (cfg.delta / 2) + offset;
                F = mvnpdf([X1(:) X2(:)],[fixation(2) fixation(1)],Sigma);
                F = reshape(F,length(x2),length(x1));
                visibility_map(ix, iy, :, :) = F;
            end
        end
        
    elseif strcmp(mode, 'flat-ellipse')
        for ix = 1:cfg.size_prior(1)
            for iy = 1:cfg.size_prior(2)
                fixation = [ix iy] * cfg.delta - (cfg.delta / 2) + offset;
                visibility_map(ix, iy, :, :) = circleFromFixation(768, 1024, cfg.size_prior, fixation(1), fixation(2), mode) .^ 8;
            end
        end
    else
        sprintf('Invalid mode for visibility map');
    end
end