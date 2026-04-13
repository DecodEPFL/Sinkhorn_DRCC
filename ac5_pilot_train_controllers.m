close all; clearvars; clc;
rng(23);

%% ==============================================================
%  AC5 TRAIN script (controller design only)
%  - Horizon: N = 10
%  - Design for multiple rho values
%  - Extended epsilon ablation for Sinkhorn
%  - Save all controllers to cache for separate OOS testing
% ===============================================================

cfg.mode = 'full';
cfg.rho_list = [0.001, 0.003, 0.007];
cfg.eps_scan_multipliers = [0.2, 0.25, 0.32, 0.4, 0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0];

cfg.tol = 1e-2;
cfg.lam_min = 0;
cfg.lam_max = 5000;

switch lower(cfg.mode)
    case 'pilot'
        cfg.eps_target_count = 4;
        cfg.out_dir = 'experiment_outputs_pilot';
    case 'full'
        cfg.eps_target_count = 8;
        cfg.out_dir = 'experiment_outputs_full';
    otherwise
        error('Unknown cfg.mode=%s. Use ''pilot'' or ''full''.', cfg.mode);
end

if ~exist(cfg.out_dir, 'dir')
    mkdir(cfg.out_dir);
end

%% Shared AC5 setup
[A, ~, B, ~, ~, ~, ~, ~, nx, ~, nu, ~, ~] = COMPleib('AC5');

sys.A = A;
sys.B = B;
sys.d = nx;
sys.p = nx;
sys.m = nu;
sys.E = eye(sys.d);
sys.eps = 1e-5;
sys.Ts = 0.1;

opt.Qt = eye(sys.d);
opt.Rt = 0.01 * eye(sys.m);
opt.N = 10;

opt.Q = kron(eye(opt.N), opt.Qt);
opt.R = kron(eye(opt.N), opt.Rt);
opt.C = blkdiag(opt.Q, opt.R);

opt.m = zeros(sys.d * opt.N, 1);
opt.Sigma_t = eye(sys.d);
opt.Sigma = kron(eye(opt.N), opt.Sigma_t);

sls.A = kron(eye(opt.N), sys.A);
sls.B = kron(eye(opt.N), sys.B);
sls.E = blkdiag(eye(sys.d), kron(eye(opt.N-1), sys.E));
sls.I = eye(sys.d * opt.N);
sls.Z = [zeros(sys.d, sys.d * (opt.N-1)) zeros(sys.d, sys.d); ...
         eye(sys.d * (opt.N-1)) zeros(sys.d * (opt.N-1), sys.d)];

state_ub = [deg2rad(20); deg2rad(15); deg2rad(10); deg2rad(30)];
H_term = [eye(sys.d); -eye(sys.d)];
b_term = [-state_ub; -state_ub];

const.H = H_term;
const.b = b_term;
const.gamma = 0.3;

%% Training noise data
opt.n = 5;
ic = [0.7; 0.1; 0.5; 0.3];
proc_noise_data = cell(1, opt.n);
for i = 1:opt.n
    proc_noise_data{i} = [ic; process_noise(opt.N-1, sys.Ts)];
end
opt.process_noise = cell2mat(proc_noise_data);

%% Train per-rho controller banks
controller_bank = struct('rho', {}, 'eps_list', {}, 'controllers', {}, 'eps_scan', {});

for ir = 1:numel(cfg.rho_list)
    rho_i = cfg.rho_list(ir);

    fprintf('\n==================================================\n');
    fprintf('Designing controllers for rho = %.6g\n', rho_i);

    [eps_list_i, eps_scan_i] = select_feasible_epsilons_near_ref( ...
        opt, rho_i, sys.eps, cfg.eps_target_count, cfg.eps_scan_multipliers);

    fprintf('Reference epsilon: %.6e\n', sys.eps);
    fprintf('Epsilon values: ');
    fprintf('%.3e ', eps_list_i);
    fprintf('\nFeasible scan points: %d / %d\n', sum(eps_scan_i.is_feasible), numel(eps_scan_i.eps));

    fprintf('Training Wasserstein and Nominal...\n');
    [~, Phi_x_wass, Phi_u_wass, phi_x_wass, phi_u_wass, ~] = ...
        train_wasserstein_controller(sys, sls, opt, const, rho_i, rho_i);
    [Phi_x_nom, Phi_u_nom, phi_x_nom, phi_u_nom, ~] = ...
        nominal_trajectory_planning(sys, sls, opt, const);

    controllers_i = struct('name', {}, 'eps_train', {}, 'Phi_x', {}, 'Phi_u', {}, 'phi_x', {}, 'phi_u', {});
    controllers_i(end+1) = mk_controller('Wasserstein', NaN, Phi_x_wass, Phi_u_wass, phi_x_wass, phi_u_wass);
    controllers_i(end+1) = mk_controller('Nominal', NaN, Phi_x_nom, Phi_u_nom, phi_x_nom, phi_u_nom);

    fprintf('Training Sinkhorn over epsilon ablation...\n');
    for ie = 1:numel(eps_list_i)
        eps_i = eps_list_i(ie);

        if ~check_radius_feasible(opt, eps_i, rho_i)
            fprintf('  [skip] eps=%.3e not feasible for rho=%.3e\n', eps_i, rho_i);
            continue;
        end

        [~, lambda_sink, Phi_x_sink, Phi_u_sink, phi_x_sink, phi_u_sink] = ...
            golden_search(rho_i, eps_i, cfg.tol, cfg.lam_min, cfg.lam_max, ...
                          sys, sls, opt, const, eps_i, rho_i);

        name = sprintf('Sinkhorn_eps%.3e', eps_i);
        controllers_i(end+1) = mk_controller(name, eps_i, Phi_x_sink, Phi_u_sink, phi_x_sink, phi_u_sink);
        fprintf('  trained %s (lambda*=%.4g)\n', name, lambda_sink);
    end

    controller_bank(end+1) = struct( ...
        'rho', rho_i, ...
        'eps_list', eps_list_i, ...
        'controllers', controllers_i, ...
        'eps_scan', eps_scan_i);
end

cache_path = fullfile(cfg.out_dir, sprintf('ac5_cached_controller_bank_N%d_%s.mat', opt.N, cfg.mode));
save(cache_path, 'controller_bank', 'cfg');

fprintf('\nTraining complete.\n');
fprintf('Controller-bank cache saved to: %s\n', cache_path);


%% ===== Local helper functions =====

function c = mk_controller(name, eps_train, Phi_x, Phi_u, phi_x, phi_u)
    c = struct('name', name, ...
               'eps_train', eps_train, ...
               'Phi_x', Phi_x, 'Phi_u', Phi_u, ...
               'phi_x', phi_x, 'phi_u', phi_u);
end

function [eps_list, scan] = select_feasible_epsilons_near_ref(opt, rho, eps_ref, target_count, scan_multipliers)
    scan.eps = unique(eps_ref .* scan_multipliers(:)');
    scan.is_feasible = false(size(scan.eps));

    for i = 1:numel(scan.eps)
        scan.is_feasible(i) = check_radius_feasible(opt, scan.eps(i), rho);
    end

    feasible_eps = scan.eps(scan.is_feasible);
    if isempty(feasible_eps)
        error(['No feasible epsilon found in local scan around eps_ref. ' ...
               'Increase cfg.eps_scan_multipliers range.']);
    end

    [~, idx] = sort(abs(log(feasible_eps ./ eps_ref)), 'ascend');
    k = min(target_count, numel(feasible_eps));
    eps_list = sort(feasible_eps(idx(1:k)));
end

function ok = check_radius_feasible(opt, epsilon, rho)
    d = size(opt.Sigma, 1);

    norm_samples = 0;
    for i = 1:size(opt.process_noise, 2)
        norm_samples = norm_samples + norm(opt.process_noise(:, i), 2)^2;
    end

    Sigma = opt.Sigma;
    m = opt.m;

    part4 = 0;
    inv_term = eye(d) + (epsilon / 2) .* inv(Sigma);

    for k = 1:opt.n
        sum_term = opt.process_noise(:, k)' * (inv_term \ opt.process_noise(:, k));
        part4 = part4 + sum_term;
    end

    part1 = -(epsilon * d / 2) * log(epsilon / 2);
    part2 = (epsilon / 2) * logdet(Sigma + (epsilon / 2) * eye(d));
    part3 = (epsilon / 2) * (m' * (Sigma \ m));
    part4 = (-part4 + norm_samples) / opt.n;

    result = part1 + part2 + part3 + part4;
    ok = (result <= rho);
end
