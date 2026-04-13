close all; clc;
rng(23);

%% ==============================================================
%  AC5 TEST script (out-of-sample only)
%  - Horizon: N = 10
%  - Tests multiple rho values
%  - Uses controller_bank from workspace OR cache
%  - Generates large OOS adversarial sets over seeds
% ===============================================================

cfg.mode = 'full';
cfg.rho_list = [];
cfg.python_cmd = '';
cfg.python_script = 'generate_adversarial_batch.py';
cfg.rho_scales = [1.1];
cfg.dist_repo_name = 'testing_distributions_rho105';

switch lower(cfg.mode)
    case 'pilot'
        cfg.seed_list = 1:3;
        cfg.n_test_particles = 100;
        cfg.out_dir = 'experiment_outputs_pilot';
    case 'full'
        cfg.seed_list = 1:100;
        cfg.n_test_particles = 100;
        cfg.out_dir = 'experiment_outputs_full';
    otherwise
        error('Unknown cfg.mode=%s. Use ''pilot'' or ''full''.', cfg.mode);
end

cfg.cache_path = fullfile(cfg.out_dir, sprintf('ac5_cached_controller_bank_N%d_%s.mat', 10, cfg.mode));
cfg.dist_repo_dir = fullfile(cfg.out_dir, cfg.dist_repo_name);

if ~exist(cfg.out_dir, 'dir')
    mkdir(cfg.out_dir);
end
if ~exist(cfg.dist_repo_dir, 'dir')
    mkdir(cfg.dist_repo_dir);
end

%% Shared AC5 setup (must match training setup)
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

opt.n = 5;
ic = [0.7; 0.1; 0.5; 0.3];
proc_noise_data = cell(1, opt.n);
for i = 1:opt.n
    proc_noise_data{i} = [ic; process_noise(opt.N-1, sys.Ts)];
end
opt.process_noise = cell2mat(proc_noise_data);

state_ub = [deg2rad(20); deg2rad(15); deg2rad(10); deg2rad(30)];
H_term = [eye(sys.d); -eye(sys.d)];
b_term = [-state_ub; -state_ub];

%% Load controller bank from workspace if available; otherwise from cache
if exist('controller_bank', 'var') == 1 && ~isempty(controller_bank)
    controller_bank_local = controller_bank;
    fprintf('Using controller_bank from MATLAB workspace (%d rho blocks).\n', numel(controller_bank_local));
else
    if ~isfile(cfg.cache_path)
        error('No controller_bank in workspace and cache not found at %s', cfg.cache_path);
    end

    S_bank = load(cfg.cache_path, 'controller_bank');
    controller_bank_local = S_bank.controller_bank;
    fprintf('Loaded controller_bank from cache: %s\n', cfg.cache_path);
end

if isempty(controller_bank_local)
    error('No controllers available for evaluation.');
end

if isempty(cfg.rho_list)
    cfg.rho_list = [controller_bank_local.rho];
    fprintf('Using rho list from cached controller bank: ');
    fprintf('%.6g ', cfg.rho_list);
    fprintf('\n');
end

%% Evaluate OOS for each rho block
results = table();

for ir = 1:numel(cfg.rho_list)
    rho_i = cfg.rho_list(ir);

    idx_bank = find(abs([controller_bank_local.rho] - rho_i) < 1e-12, 1, 'first');
    if isempty(idx_bank)
        warning('No controller block found for rho=%.6g. Skipping this rho.', rho_i);
        continue;
    end

    controllers_i = controller_bank_local(idx_bank).controllers;
    eps_list_i = controller_bank_local(idx_bank).eps_list;

    if isempty(controllers_i) || isempty(eps_list_i)
        warning('Empty controller block for rho=%.6g. Skipping.', rho_i);
        continue;
    end

    validate_controller_dims(controllers_i, sys, opt);

    rho_tag = sprintf('rho_%.6g', rho_i);
    adv_dir_i = fullfile(cfg.dist_repo_dir, ['adversarial_sets_' rho_tag]);
    if ~exist(adv_dir_i, 'dir')
        mkdir(adv_dir_i);
    end

    nominal_path_i = fullfile(cfg.out_dir, sprintf('nominal_N%d_%s.mat', opt.N, rho_tag));
    X = opt.process_noise;
    save(nominal_path_i, 'X');

    eps_csv = join(string(eps_list_i), ',');
    rho_scales_csv = join(string(cfg.rho_scales), ',');
    seeds_csv = join(string(cfg.seed_list), ',');

    cmd = sprintf([ ...
        '%s %s --nominal "%s" --out-dir "%s" --n-particles %d ' ...
        '--rho %.12g --eps-list "%s" --rho-scales "%s" --seeds "%s"'], ...
        resolve_python_command(cfg.python_cmd), cfg.python_script, nominal_path_i, adv_dir_i, cfg.n_test_particles, ...
        rho_i, eps_csv{1}, rho_scales_csv{1}, seeds_csv{1});

    fprintf('\nGenerating OOS adversarial sets for rho=%.6g:\n%s\n\n', rho_i, cmd);
    status = system(cmd);
    if status ~= 0
        warning('Python adversarial generation failed for rho=%.6g (status=%d). Skipping.', rho_i, status);
        continue;
    end

    manifest_path = fullfile(adv_dir_i, 'manifest.csv');
    if ~isfile(manifest_path)
        warning('Manifest not found for rho=%.6g at %s. Skipping.', rho_i, manifest_path);
        continue;
    end
    manifest = readtable(manifest_path, 'Delimiter', ',');

    fprintf('Evaluating %d controllers on %d OOS datasets for rho=%.6g...\n', ...
        numel(controllers_i), height(manifest), rho_i);

    for r = 1:height(manifest)
        eps_test = manifest.eps(r);
        S = load(manifest.file{r});

        for c = 1:numel(controllers_i)
            eps_train = controllers_i(c).eps_train;
            is_baseline = isnan(eps_train);
            is_matching_sinkhorn = ~is_baseline && abs(eps_train - eps_test) < 1e-15;

            if ~(is_baseline || is_matching_sinkhorn)
                continue;
            end

            expected_w_dim = size(controllers_i(c).Phi_x, 2);
            W_test = adapt_disturbance_matrix(S.X_test, expected_w_dim, manifest.file{r});

            m = evaluate_controller( ...
                controllers_i(c).Phi_x, controllers_i(c).Phi_u, ...
                controllers_i(c).phi_x, controllers_i(c).phi_u, ...
                W_test, opt, H_term, b_term, sys.d);

            row = table( ...
                rho_i, eps_test, manifest.seed(r), string(controllers_i(c).name), eps_train, ...
                m.mean_cost, m.violation_rate, ...
                'VariableNames', {'rho_train','eps_ablation','seed','controller','eps_train','mean_cost','violation_rate'});

            results = [results; row];
        end
    end
end

if isempty(results)
    error('No out-of-sample ablation results produced.');
end

summary = groupsummary(results, {'rho_train','eps_ablation','controller'}, {'mean','std'}, {'mean_cost','violation_rate'});

best_sinkhorn = summarize_best_sinkhorn(results);

results_path = fullfile(cfg.out_dir, sprintf('ac5_oos_ablation_results_%s.csv', cfg.mode));
summary_path = fullfile(cfg.out_dir, sprintf('ac5_oos_ablation_summary_%s.csv', cfg.mode));
best_sinkhorn_path = fullfile(cfg.out_dir, sprintf('ac5_oos_best_sinkhorn_vs_baselines_%s.csv', cfg.mode));

writetable(results, results_path);
writetable(summary, summary_path);
writetable(best_sinkhorn, best_sinkhorn_path);

fprintf('\nDone.\n');
fprintf('Full results : %s\n', results_path);
fprintf('Summary      : %s\n', summary_path);
fprintf('Best Sinkhorn: %s\n', best_sinkhorn_path);


%% ===== Local helper functions =====

function metrics = evaluate_controller(Phi_x, Phi_u, phi_x, phi_u, W, opt, H, b, d)
    n = size(W, 2);
    costs = zeros(n, 1);
    violations = false(n, 1);

    for i = 1:n
        wi = W(:, i);
        x_stack = Phi_x * wi + phi_x;
        u_stack = Phi_u * wi + phi_u;

        xu = [x_stack; u_stack];
        costs(i) = xu' * opt.C * xu;

        xN = x_stack(end-d+1:end);
        residual = H * xN + b;
        violations(i) = any(residual > 0);
    end

    metrics.mean_cost = mean(costs);
    metrics.violation_rate = mean(violations);
end

function W_out = adapt_disturbance_matrix(W_in, expected_dim, src_file)
    [r, c] = size(W_in);
    if r == expected_dim
        W_out = W_in;
        return;
    end
    if c == expected_dim
        W_out = W_in';
        return;
    end

    error(['Disturbance dimension mismatch for %s. Loaded size is %dx%d, expected one dimension to be %d. ' ...
           'This usually means controllers and nominal/OOS data were built with different horizon N.'], ...
          src_file, r, c, expected_dim);
end

function validate_controller_dims(controllers_local, sys, opt)
    expected_w_dim = sys.d * opt.N;
    expected_x_dim = sys.d * opt.N;
    expected_u_dim = sys.m * opt.N;

    for i = 1:numel(controllers_local)
        px = controllers_local(i).Phi_x;
        pu = controllers_local(i).Phi_u;

        if size(px, 1) ~= expected_x_dim || size(px, 2) ~= expected_w_dim || ...
           size(pu, 1) ~= expected_u_dim || size(pu, 2) ~= expected_w_dim
            error(['Controller "%s" is incompatible with current setup (N=%d). ' ...
                   'Expected Phi_x [%d x %d], Phi_u [%d x %d], got Phi_x [%d x %d], Phi_u [%d x %d]. ' ...
                   'Use controllers trained with the same N (or clear stale workspace variables).'], ...
                  controllers_local(i).name, opt.N, ...
                  expected_x_dim, expected_w_dim, expected_u_dim, expected_w_dim, ...
                  size(px, 1), size(px, 2), size(pu, 1), size(pu, 2));
        end
    end
end

function out = summarize_best_sinkhorn(results)
    rho_vals = unique(results.rho_train);
    out = table();

    for ir = 1:numel(rho_vals)
        rho_i = rho_vals(ir);
        sub = results(results.rho_train == rho_i, :);

        sink = sub(sub.controller == "Sinkhorn" | startsWith(sub.controller, "Sinkhorn_") | startsWith(sub.controller, "Sinkhorn"), :);
        if isempty(sink)
            sink = sub(startsWith(sub.controller, "Sinkhorn"), :);
        end
        wass = sub(sub.controller == "Wasserstein", :);
        nom = sub(sub.controller == "Nominal", :);

        sink_summary = groupsummary(sink, 'eps_ablation', {'mean','std'}, {'mean_cost','violation_rate'});
        if isempty(sink_summary)
            continue;
        end

        sink_summary = sortrows(sink_summary, {'mean_violation_rate','mean_mean_cost'});
        best_eps = sink_summary.eps_ablation(1);
        best_row = sink_summary(1, :);

        wass_cost = mean(wass.mean_cost);
        wass_viol = mean(wass.violation_rate);
        nom_cost = mean(nom.mean_cost);
        nom_viol = mean(nom.violation_rate);

        row = table( ...
            rho_i, best_eps, ...
            best_row.mean_mean_cost, best_row.mean_violation_rate, ...
            wass_cost, wass_viol, ...
            nom_cost, nom_viol, ...
            wass_cost - best_row.mean_mean_cost, ...
            wass_viol - best_row.mean_violation_rate, ...
            nom_cost - best_row.mean_mean_cost, ...
            nom_viol - best_row.mean_violation_rate, ...
            'VariableNames', { ...
                'rho_train','best_sinkhorn_eps', ...
                'sinkhorn_mean_cost','sinkhorn_violation_rate', ...
                'wasserstein_mean_cost','wasserstein_violation_rate', ...
                'nominal_mean_cost','nominal_violation_rate', ...
                'wasserstein_cost_minus_sinkhorn', ...
                'wasserstein_violation_minus_sinkhorn', ...
                'nominal_cost_minus_sinkhorn', ...
                'nominal_violation_minus_sinkhorn'});

        out = [out; row];
    end
end

function py_cmd = resolve_python_command(user_cmd)
    candidates = {};

    if ~isempty(user_cmd)
        candidates{end+1} = user_cmd;
    end

    env_python = getenv('PYTHON_BIN');
    if ~isempty(env_python)
        candidates{end+1} = env_python;
    end

    candidates{end+1} = 'python3';
    candidates{end+1} = 'python';

    for i = 1:numel(candidates)
        cand = strtrim(candidates{i});
        if isempty(cand)
            continue;
        end

        cand_q = shell_quote(cand);
        [status, ~] = system(sprintf('%s -c "import sys; print(sys.version)"', cand_q));
        if status == 0
            py_cmd = cand_q;
            return;
        end
    end

        error(['No working Python executable found. Set cfg.python_cmd explicitly, e.g. ' ...
            'cfg.python_cmd = ''/path/to/python'' or set PYTHON_BIN in your environment.']);
end

function q = shell_quote(s)
    s = strtrim(s);
    if startsWith(s, '"') && endsWith(s, '"')
        q = s;
        return;
    end
    q = sprintf('"%s"', s);
end
