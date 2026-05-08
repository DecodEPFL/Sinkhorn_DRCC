close all; clc;
rng(42);

% Build ac5_iid_plot_data.mat from the i.i.d. OOS results.
% Mirrors ac5_build_plot_data.m but uses fresh i.i.d. noise instead of
% adversarial samples.
%
% Then run:
%   python plot_ac5_iid_results.py

cfg.out_dir    = 'experiment_outputs_iid';
cfg.cache_path = fullfile('experiment_outputs_full', ...
                          'ac5_cached_controller_bank_N10_full.mat');
cfg.rho_target = 0.007;     % choose one among [0.001, 0.003, 0.007]
cfg.n_plot     = 200;       % number of realisations to include in the plot
cfg.output_mat = fullfile(cfg.out_dir, 'ac5_iid_plot_data.mat');

% Auto-discover the latest iid_oos_results CSV to pick the best Sinkhorn eps
results_files = dir(fullfile(cfg.out_dir, 'iid_oos_results_*.csv'));
if isempty(results_files)
    error(['No iid_oos_results_*.csv found in %s. ' ...
           'Run ac5_iid_oos_test.m first.'], cfg.out_dir);
end
[~, idx] = max([results_files.datenum]);
cfg.results_csv = fullfile(cfg.out_dir, results_files(idx).name);
fprintf('Using results: %s\n', cfg.results_csv);

%% ------------------------------------------------------------------
%  System setup  (must match training)
% ------------------------------------------------------------------
[A, ~, B, ~, ~, ~, ~, ~, nx, ~, nu, ~, ~] = COMPleib('AC5');
sys.A  = A;  sys.B = B;
sys.d  = nx; sys.m = nu;
sys.Ts = 0.1;
opt.N  = 10;

ic       = [0.7; 0.1; 0.5; 0.3];
state_ub = [deg2rad(20); deg2rad(15); deg2rad(10); deg2rad(30)];

%% ------------------------------------------------------------------
%  Load controller bank
% ------------------------------------------------------------------
if ~isfile(cfg.cache_path)
    error('Cache not found: %s', cfg.cache_path);
end
S = load(cfg.cache_path, 'controller_bank');
controller_bank = S.controller_bank;

idx_block = find(abs([controller_bank.rho] - cfg.rho_target) < 1e-12, 1);
if isempty(idx_block)
    error('No block found for rho=%.6g', cfg.rho_target);
end
block = controller_bank(idx_block);

%% ------------------------------------------------------------------
%  Pick best Sinkhorn eps from the i.i.d. results
%  (lowest violation_rate, tie-broken by mean_cost)
% ------------------------------------------------------------------
tbl = readtable(cfg.results_csv);
tbl_rho = tbl(abs(tbl.rho_train - cfg.rho_target) < 1e-12, :);

sink_rows = tbl_rho(startsWith(tbl_rho.controller, 'Sinkhorn'), :);
sink_rows = sortrows(sink_rows, {'violation_rate', 'mean_cost'}, {'ascend','ascend'});
best_eps  = sink_rows.eps_train(1);
fprintf('Best Sinkhorn eps for rho=%.6g: %.3e\n', cfg.rho_target, best_eps);

%% ------------------------------------------------------------------
%  Locate the three controllers
% ------------------------------------------------------------------
names_bank = {block.controllers.name};
eps_bank   = [block.controllers.eps_train];

idx_sink = find(startsWith(string(names_bank), 'Sinkhorn') & ...
                abs(eps_bank - best_eps) < 1e-15, 1);
idx_wass = find(strcmp(names_bank, 'Wasserstein'), 1);
idx_nom  = find(strcmp(names_bank, 'Nominal'), 1);

if isempty(idx_sink) || isempty(idx_wass) || isempty(idx_nom)
    error('Could not find all three controllers for rho=%.6g', cfg.rho_target);
end

ctrl_idx         = [idx_sink, idx_wass, idx_nom];
controller_names = {'Sinkhorn', 'Wasserstein', 'Empirical'};
Phi_x_list       = cell(1, 3);
phi_x_list       = cell(1, 3);
for k = 1:3
    Phi_x_list{k} = block.controllers(ctrl_idx(k)).Phi_x;
    phi_x_list{k} = block.controllers(ctrl_idx(k)).phi_x;
end

%% ------------------------------------------------------------------
%  Generate i.i.d. noise realisations
% ------------------------------------------------------------------
w_dim  = sys.d * opt.N;
W_plot = zeros(w_dim, cfg.n_plot);
for k = 1:cfg.n_plot
    W_plot(:, k) = [ic; process_noise(opt.N - 1, sys.Ts)];
end

%% ------------------------------------------------------------------
%  Export
% ------------------------------------------------------------------
if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
export_ac5_plot_data(cfg.output_mat, controller_names, Phi_x_list, phi_x_list, ...
                     W_plot, sys, opt, state_ub);

fprintf('\nBuilt %s  (rho=%.6g, best eps=%.3e, n_plot=%d)\n', ...
        cfg.output_mat, cfg.rho_target, best_eps, cfg.n_plot);
fprintf('Now run: python plot_ac5_iid_results.py\n');
