function expand_info(dir_data)
    % EXPAND_INFO Add new fields to "info" with useful information
    if nargin == 0
        dir_data = './experiment_data/';
    end

    info = [];
    tmp = dir([dir_data 'info/' '*_info.mat']); filenames = {tmp.name};

    for su=1:length(filenames)
        load([dir_data 'info/' filenames{su}]);
        for i = 1:length(info)
            tr = info(i).tr;
            if isempty(exp.response(tr).size)
                continue;
            end

            exp_response = exp.response(tr);
            exp_response.position = exp_response.position - [128 128];

            % put all data from exp in info
            info(i).exp_response = exp_response;
            info(i).exp_data     = exp.data(tr);
            info(i).exp_trial    = exp.trial(tr);
            info(i).subj         = exp.subj.subject;

            % did we really find the target, besides what stopcriteria says?
            % we accept a 11 pixel distance to the actual target, which is 25%
            % of the target size
            center = [(info(i).matched_column + info(i).template_side_length / 2)
                        (info(i).matched_row + info(i).template_side_length / 2)]';
            info(i).target_center      = center;
            fix2target                 = info(i).fixations - repmat(center, size(info(i).fixations, 1), 1);
            info(i).dist_allfix2target = sqrt(sum(abs(fix2target).^2, 2)); % dist to target center
            info(i).target_found       = (any(max(abs(fix2target)')' < info(i).template_side_length / 2 * 1.25));

            % add distance from all fix to center, and from guess to center
            guess                      = exp_response.position;
            fix2guess                  = info(i).fixations - repmat(guess, size(info(i).fixations, 1), 1);
            info(i).dist_allfix2guess  = sqrt(sum(abs(fix2guess).^2, 2));

            info(i).dist_guess2target  = norm(guess-center);
        end
        save([dir_data 'info/' filenames{su}], 'info', 'exp');
    end
end