function export_ac5_plot_data(out_file, controller_names, Phi_x_list, phi_x_list, W, sys, opt, state_ub)
% Export AC5 trajectory data for Python plotting.
%
% Inputs
%   out_file          : output .mat file path
%   controller_names  : cell array of controller names, e.g. {'Sinkhorn','Wasserstein','Nominal'}
%   Phi_x_list        : cell array of Phi_x matrices
%   phi_x_list        : cell array of phi_x vectors
%   W                 : disturbance samples matrix (sys.d*opt.N x n_samples)
%   sys, opt          : system and optimization structs
%   state_ub          : (sys.d x 1) terminal state upper bounds
%
% Output fields in .mat
%   trajectories      : [n_ctrl, n_samples, N, d]
%   controller_names  : cell array
%   state_ub          : [d,1]
%   time_idx          : [N,1]

    n_ctrl = numel(controller_names);
    n_samples = size(W, 2);
    N = opt.N;
    d = sys.d;

    trajectories = zeros(n_ctrl, n_samples, N, d);

    for c = 1:n_ctrl
        Phi_x = Phi_x_list{c};
        phi_x = phi_x_list{c};

        for i = 1:n_samples
            wi = W(:, i);
            x_stack = Phi_x * wi + phi_x;
            x_mat = reshape(x_stack, d, N)'; % [N, d]
            trajectories(c, i, :, :) = x_mat;
        end
    end

    time_idx = (1:N)';

    save(out_file, 'trajectories', 'controller_names', 'state_ub', 'time_idx');
    fprintf('Saved AC5 plotting data to %s\n', out_file);
end
