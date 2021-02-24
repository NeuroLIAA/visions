function [errorPatch,dataLine] = niceBars(varargin)
% NICEBARS plots transparent patch errorbars around a given curve
%
%     [errorPatch,dataLine,t] = NICEBARS(X, Y, LCI, UCI, Color, Alpha)
%     Plots a transperent error patches around a line. Upper and lower
%     error bars are independent. Color and alpha are optional arguments.
%
%     USAGE :
%              1)   [errorPatch,dataLine] = niceBars(x,y,CI)
%                    Calculates the Lower and upper errorbars as
%                    L = Y-E and U = Y+E. It then takes a default color
%                    as patch color, and the line color as black and plots
%                    it around the line using a patch object.
%
%              2)   [errorPatch,dataLine] = niceBars(x,y,LCI, UCI)
%                    Calculates the Lower and upper errorbars as
%                    L = Y-E and U = Y+E.
%
%              3)   [errorPatch,dataLine] = niceBars(x,y,CI,color,alpha)
%                    User single error CI, It then takes a custom color and
%                    alpha value.
%
%              4)   [errorPatch,dataLine] = niceBars(x,y,LCI,UCI,color,alpha)
%                    User Supplied bounds are taken as LCI and UCI, It then
%                   takes a custom color and alpha value.
%
%     NOTE : X, Y, LCI, UCI or CI must be row vectors
%
%      EXAMPLE
%                         t = [-5:0.05:5];
%                         Y = sin(t);
%                         E = 0.4*rand(1,length(t));
%                         [errorPatch,dataLine] = niceBars(t,Y,E);
%                         xlabel('time');
%                         ylabel('Amplitude');
%                         title('Using Errors alone');
%
% Version 1.0 Bob Colner April Fools Day 2009

switch(nargin)
    case 3,
        % If there are 3 inputs it means its just the errors
        x = varargin{1};
        y = varargin{2};
        E = varargin{3};
        L = y - E;
        U = y + E;
        color = [0 0 0];
        alpha = 0.5;
    case 4,
        % If there are 4 inputs it means they entered the Lower and upper
        % bounds
        x = varargin{1};
        y = varargin{2};
        L = varargin{3};
        U = varargin{4};
        color = [0 0 0];
        alpha = 0.5;
    case 5,
        % equal error bars and custom color and alpha
        x = varargin{1};
        y = varargin{2};
        E = varargin{3};
        L = y - E;
        U = y + E;
        color =  varargin{4};
        alpha = varargin{5};
    case 6,
        % independent error bars and custom color and alpha
        x = varargin{1};
        y = varargin{2};
        L = varargin{3};
        U = varargin{4};
        color =  varargin{5};
        alpha = varargin{6};
end
Xcoords = [x x(end:-1:1)];% X-coords for CI patch 
Ycoords = [U L(end:-1:1)];% Y-coords for CI patch 

hold on;
    errorPatch = patch(Xcoords,Ycoords,color);% draw error bar patch
%     dataLine = plot(x,y,'color',color,'linewidth',2);% plot data line
hold off;

% set(errorPatch,'linestyle','none','linewidth',2,'faceAlpha',alpha);% set 'errorPatch' style
% set(get(get(errorPatch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off'); % Exclude line from legend

set(gcf, 'color', 'white')% set figure backgroup to white
set(gca, 'fontsize', 14)% set axis font size
