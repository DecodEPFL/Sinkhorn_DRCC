close all; clc;

% Build ac5_plot_data.mat directly from cached full experiment outputs.
% Then run:
%   python plot_ac5_trajectories.py --input ac5_plot_data.mat --out-dir ac5_figures

cfg.out_dir = 'experiment_outputs_full';
cfg.cache_path = fullfile(cfg.out_dir, 'ac5_cached_controller_bank_N10_full.mat');
cfg.best_csv = fullfile(cfg.out_dir, 'ac5_oos_best_sinkhorn_vs_baselines_full.csv');
cfg.rho_target = 0.007;    % choose one among [0.001, 0.003, 0.007]
cfg.seed = 1;              % realization to visualize
cfg.output_mat = 'ac5_plot_data.mat';

if ~isfile(cfg.cache_path)
    error('Cached controller bank not found: %s', cfg.cache_path);
end
if ~isfile(cfg.best_csv)
    error('Best Sinkhorn summary CSV not found: %s', cfg.best_csv);
end

S = load(cfg.cache_path, 'controller_bank');
controller_bank = S.controller_bank;

idx_block = find(abs([controller_bank.rho] - cfg.rho_target) < 1e-12, 1, 'first');
if isempty(idx_block)
    error('No controller block found for rho=%.6g in cache.', cfg.rho_target);
end
block = controller_bank(idx_block);

best_tbl = readtable(cfg.best_csv);
idx_best = find(abs(best_tbl.rho_train - cfg.rho_target) < 1e-12, 1, 'first');
if isempty(idx_best)
    error('No best-Sinkhorn row found in %s for rho=%.6g', cfg.best_csv, cfg.rho_target);
end
best_eps = best_tbl.best_sinkhorn_eps(idx_best);

idx_sink = find(startsWith(string({block.controllers.name}), 'Sinkhorn') & ...
                abs([block.controllers.eps_train] - best_eps) < 1e-15, 1, 'first');
idx_wass = find(strcmp({block.controllers.name}, 'Wasserstein'), 1, 'first');
idx_nom  = find(strcmp({block.controllers.name}, 'Nominal'), 1, 'first');

if isempty(idx_sink) || isempty(idx_wass) || isempty(idx_nom)
    error('Could not find Sinkhorn/Wasserstein/Nominal controllers for rho=%.6g', cfg.rho_target);
end

ctrl_idx = [idx_sink, idx_wass, idx_nom];
controller_names = {'Sinkhorn', 'Wasserstein', 'Nominal'};
Phi_x_list = cell(1, numel(ctrl_idx));
phi_x_list = cell(1, numel(ctrl_idx));
for k = 1:numel(ctrl_idx)
    Phi_x_list{k} = block.controllers(ctrl_idx(k)).Phi_x;
    phi_x_list{k} = block.controllers(ctrl_idx(k)).phi_x;
end

adv_dir = fullfile(cfg.out_dir, sprintf('adversarial_sets_rho_%.6g', cfg.rho_target));
manifest_path = fullfile(adv_dir, 'manifest.csv');
if ~isfile(manifest_path)
    error('Manifest not found: %s', manifest_path);
end
manifest = readtable(manifest_path);

mask = abs(manifest.eps - best_eps) < 1e-15 & manifest.seed == cfg.seed;
idx_row = find(mask, 1, 'first');
if isempty(idx_row)
    error('No adversarial sample found for rho=%.6g, eps=%.6g, seed=%d', cfg.rho_target, best_eps, cfg.seed);
end

Sx = load(manifest.file{idx_row}, 'X_test');
W = Sx.X_test;

expected_w_dim = size(Phi_x_list{1}, 2);
if size(W,1) ~= expected_w_dim && size(W,2) == expected_w_dim
    W = W';
elseif size(W,1) ~= expected_w_dim
    error('X_test shape mismatch: got %dx%d, expected one dimension=%d', size(W,1), size(W,2), expected_w_dim);
end

[A, ~, B, ~, ~, ~, ~, ~, nx, ~, nu, ~, ~] = COMPleib('AC5');
sys.A = A;
sys.B = B;
sys.d = nx;
sys.m = nu;
opt.N = 10;

state_ub = [deg2rad(20); deg2rad(15); deg2rad(10); deg2rad(30)];

export_ac5_plot_data(cfg.output_mat, controller_names, Phi_x_list, phi_x_list, W, sys, opt, state_ub);

fprintf('\nBuilt %s using rho=%.6g, best eps=%.6g, seed=%d\n', cfg.output_mat, cfg.rho_target, best_eps, cfg.seed);
fprintf('Now run: python plot_ac5_trajectories.py --input %s --out-dir ac5_figures\n', cfg.output_mat);
