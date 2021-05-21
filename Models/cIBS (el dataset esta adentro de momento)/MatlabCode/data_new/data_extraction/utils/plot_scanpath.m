% Plot scanpath with target and background image
function plot_scanpath(info, exp, tr, subject_name)
    % info, exp from X_info.mat
    % tr = #trial
    % fixations = array of fixations (N x 2, being N = #fixations)
    if nargin < 4
        subject_name = 'Undefined';
    end

    %image_folder = '../VisualSearch_Interiores/images_for_experiment/';
    image_folder = './';

    plotstyle = {'o-', 'LineWidth', 2, 'MarkerEdgeColor', 'y', 'MarkerFaceColor', 'r', 'markersize', 8};
    
    figure(tr)
    hold on
        % title
        h = title({['Image: ' exp.trial(tr).imgname ', found: ' num2str(exp.data(tr).stopcriteria == 2)]; ...
          ['Subject: ' subject_name]});
        set(h,'interpreter','none')

        % background image
        colormap('gray')
        imagesc(imread([image_folder exp.trial(tr).imgname])) %flip

        % target position
        rectangle('Position', [info(tr).matched_column, info(tr).matched_row, ...
                              info(tr).template_side_length, info(tr).template_side_length], 'EdgeColor', 'r')

        % fixation plotting 
        plot(info.fixations(:,1), info.fixations(:,2), plotstyle{:})
        scatter(info.fixations(end,1), info.fixations(end,2),     'MarkerEdgeColor',[0 .5 .5],...
                                                                  'MarkerFaceColor',[0 .7 .7],...
                                                                  'LineWidth',1.5)
        set(gca,'ydir','reverse')

        axis([0 1024 0 768])
    hold off
end