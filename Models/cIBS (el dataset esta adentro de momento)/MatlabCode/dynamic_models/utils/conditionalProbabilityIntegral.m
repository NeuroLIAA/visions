function result = conditionalProbabilityIntegral(ix, iy, kx, ky, T, p, visibility_map, alpha)
%    m = (-2*log(p(:,:,T) / p(ix,iy,T)) + visibility_map(:,:,kx,ky).^2 + visibility_map(ix,iy,kx,ky).^2) ./ (2 * visibility_map(:,:,kx,ky)); % MAL
%    b =  visibility_map(ix,iy,kx,ky) ./  visibility_map(:,:,kx,ky); % MAL

    b = (-2*log(p(:,:,T) / p(ix,iy,T)) + visibility_map(:,:,kx,ky).^2 + visibility_map(ix,iy,kx,ky).^2) ./ (2 * visibility_map(:,:,kx,ky)); % BIEN
    m =  visibility_map(ix,iy,kx,ky) ./  visibility_map(:,:,kx,ky); % BIEN

    % we ensure that the product is only for i != j (normcdf(1000000) = 1)
    m(ix, iy) = 0;
    b(ix, iy) = 1000000;

    % check limits to integrate (normcdf(-20) = 0 and so will be the product)
    minw = max(max((-20 - b(m>0)) ./ m(m>0)), -20);
    if isempty(minw)
        minw = -20;
    end
    maxw = min(min((-20 - b(m<0)) ./ m(m<0)), 20);
    if isempty(maxw)
        maxw = 20;
    end
        
    if minw >= maxw
%        [ix iy minw maxw] 
        result = 0;
        return;
    end
    
    num = 50;
    wRange = linspace(minw, maxw, num);
    
    tmp = m(:) * wRange + repmat(b(:), 1, length(wRange));
    tmp(isnan(tmp)) = 1;

%    phiW = exp(-0.5 .* wRange .* wRange / sqrt(2 * pi)); % MAL
    phiW = exp(-0.5 .* wRange .* wRange) / sqrt(2 * pi); % BIEN
    point = phiW .* (prod(alpha * normcdf(tmp), 1) / alpha);

    result = trapz(wRange, point);
end
