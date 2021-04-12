function mappedFix = mapToReducedMatrix(fixation, delta, image_size)
    % mapToReducedMatrix: maps a pixel in the image to a cell in de grid
    
    % Input:
    % fixation     = pixel in the image
    % delta        = Size of grid cells
    % image_size   = Size of the image
    
    grid_size = floor(image_size / delta);
    offset = floor((image_size - grid_size*delta) / 2);
    
    mappedFix = ceil((double(fixation) - offset) / delta);
end