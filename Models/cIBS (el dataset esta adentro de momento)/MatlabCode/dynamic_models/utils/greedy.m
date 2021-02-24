function [idx_x, idx_y] = greedy(p,T)
    [~, idx_y] = max(max(p(:,:,T)));
    [~, idx_x] = max(p(:,idx_y,T));
end
