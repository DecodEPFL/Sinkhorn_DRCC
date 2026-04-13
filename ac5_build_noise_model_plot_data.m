close all; clc;

% Build plotting data for the noise-model perturbation benchmark:
%   1) Best Sinkhorn (from noise-model best CSV)
%   2) Wasserstein
%   3) Nominal (Empirical)
%
% Output MAT can be consumed by plot_ac5_noise_model_results.py to produce
% terminal in-box scatter plots at the last time step.

cfg.mode = 'full';
cfg.horizon = 10;
cfg.n_test_particles = 100;
cfg.out_dir = 'experiment_outputs_noise_model_full_winner_guided_small_rho_r200';
cfg.cache_dir = 'experiment_outputs_full';

cfg.cache_path = fullfile(cfg.cache_dir, sprintf('ac5_cached_controller_bank_N%d_full.mat', cfg.horizon));
cfg.best_csv = fullfile(cfg.out_dir, sprintf('ac5_noise_model_best_sinkhorn_vs_baselines_%s.csv', cfg.mode));
cfg.scenarios_csv = fullfile(cfg.out_dir, sprintf('ac5_noise_model_perturbation_scenarios_%s.csv', cfg.mode));

cfg.rho_target = 0.007;
cfg.seed_to_plot = 1; % set [] to auto-pick first available seed for this rho
cfg.output_mat = fullfile(cfg.out_dir, 'ac5_noise_model_plot_data.mat');

% Tip:
% - If you run a different benchmark folder, only update cfg.out_dir.
% - If controllers are in a custom path, update cfg.cache_path.

if ~isfile(cfg.cache_path)
    error('Cached controller bank not found: %s', cfg.cache_path);
end
if ~isfile(cfg.best_csv)
    error('Best-Sinkhorn CSV not found: %s', cfg.best_csv);
end
if ~isfile(cfg.scenarios_csv)
    error('Scenario CSV not found: %s', cfg.scenarios_csv);
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

scenarios = readtable(cfg.scenarios_csv);
scenarios = scenarios(abs(scenarios.rho_train - cfg.rho_target) < 1e-12, :);
if isempty(scenarios)
    error('No scenario rows for rho=%.6g in %s', cfg.rho_target, cfg.scenarios_csv);
end

if ~ismember('seed', scenarios.Properties.VariableNames)
    error('Expected a seed column in scenarios CSV: %s', cfg.scenarios_csv);
end

if isempty(cfg.seed_to_plot)
    cfg.seed_to_plot = scenarios.seed(1);
end

scenarios = scenarios(scenarios.seed == cfg.seed_to_plot, :);
if isempty(scenarios)
    error('No scenario row found for rho=%.6g and seed=%d in %s', ...
        cfg.rho_target, cfg.seed_to_plot, cfg.scenarios_csv);
end

% System setup for export
[A, ~, B, ~, ~, ~, ~, ~, nx, ~, nu, ~, ~] = COMPleib('AC5');
sys.A = A;
sys.B = B;
sys.d = nx;
sys.m = nu;
opt.N = cfg.horizon;

state_ub = [deg2rad(20); deg2rad(15); deg2rad(10); deg2rad(30)];
ic = [0.7; 0.1; 0.5; 0.3];
Ts = 0.1;

n_scen = height(scenarios);
n_total = n_scen * cfg.n_test_particles;
w_dim = sys.d * opt.N;
W = zeros(w_dim, n_total);

fprintf('Building disturbance matrix from 1 scenario (seed=%d) x %d particles...\n', ...
    cfg.seed_to_plot, cfg.n_test_particles);
col = 1;
for r = 1:n_scen
    model = struct();
    model.V_tas = scenarios.V_tas(r);
    model.b = scenarios.b(r);
    model.L_v = scenarios.L_v(r);
    model.L_w = scenarios.L_w(r);
    model.sigma_v = scenarios.sigma_v(r);
    model.sigma_w = scenarios.sigma_w(r);
    model.shared_latent_input = logical(scenarios.shared_latent_input(r));

    for i = 1:cfg.n_test_particles
        W(:, col) = [ic; process_noise(opt.N-1, Ts, model)];
        col = col + 1;
    end
end

export_ac5_plot_data(cfg.output_mat, controller_names, Phi_x_list, phi_x_list, W, sys, opt, state_ub);

fprintf('\nBuilt %s\n', cfg.output_mat);
fprintf('rho=%.6g, best eps=%.6g, seed=%d, total samples=%d\n', ...
    cfg.rho_target, best_eps, cfg.seed_to_plot, n_total);
