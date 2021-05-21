function reduced = reduceMatrix (saliencyMap, delta, mode)
    % reduceMatrix: creates a grid on the saliency map where each cell measures 
    % delta by delta. In the case that there are pixels that do not  
    % complete a cell, the grid will be center in the matrix and the pixels 
    % that are not in them will be discarded. 
    % The function returns a reduced matrix which, in each position, 
    % contains the maximum or mean value (depending on the mode 
    % parameter) of the pixels of each cell in the grid.
    % Note: if matrix is divisible by delta, the first cell goes from 1 to 
    % delta, the next one from delta + 1 to 2 * delta.
    
    % Input:
    % saliencyMap = Saliency map  
    % delta       = Cell size
    % mode        = Value of the cell: 'mean' | 'max'
    
    dims = size(saliencyMap);
    reduced = nan(floor(dims / delta));
    offset = floor((dims - size(reduced)*delta) / 2);
    
    for i=1:size(reduced,1)
        for j=1:size(reduced,2)
            begin = ([i-1 j-1] * delta) + 1 + offset;
            endd  = begin + delta -1;
            
            if strcmp(mode, 'mean')
                reduced(i, j) = mean2(saliencyMap(begin(1):endd(1), begin(2):endd(2)));
            elseif strcmp(mode, 'max')
                reduced(i, j) = max(max(saliencyMap(begin(1):endd(1), begin(2):endd(2))));
            else
                sprintf('Invalid mode for reducing matrix');
            end
        end
    end
end