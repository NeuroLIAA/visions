%rainbow_colors    [C] = rainbow_colors(n)
%
% returns a set of n colors following a
% stretch of the rainbow.

% (c) 2004 CK Machens & CD Brody

function [C] = rainbow_colors(nclasses)
   
   rainbowcolormap = hsv(256); 
   rainbowcolormap = ...
       rainbowcolormap([250:256 1:40 50:110 135:155],:); 
   rainbowcolormap = rainbowcolormap(end:-1:1,:);
   cmap = rainbowcolormap;

   C = zeros(nclasses, 3);
   
   for i=1:nclasses,
      g = ((i-1)/(nclasses-1))*(size(cmap,1)-1) + 1;
      g = round(g);
      C(i,:) = cmap(g,:);
   end;
