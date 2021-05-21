structured_mat = load('initial_fixations.mat');
initial_fixations = [structured_mat.initial_fixations{:}];
save('initial_fixations.mat', 'initial_fixations');

structured_mat = load('target_positions_filtered.mat');
target_positions = [structured_mat.target_positions{:}];
save('target_positions_filtered.mat', 'target_positions');

