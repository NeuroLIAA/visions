% s = fun_xy2fixations(s)
%
% If s is a structure, this function adds the field 'fixations' as [x y]
% If s is a 2xN matrix, it's the same as defining fixation = [x y];

function s = fun_xy2fixations(s)
    if isstruct(s) 
        for tr = 1:length(s)
            if size(s(tr).x,1) < size(s(tr).x,2)
                s(tr).fixations = [s(tr).x' s(tr).y'];
            else
                s(tr).fixations = [s(tr).x s(tr).y];
            end
        end
    else
        % Nada por ahora
    end
end