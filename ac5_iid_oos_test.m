close all; clc;
rng(42);

%% =======================================================================
%  AC5 i.i.d. OUT-OF-SAMPLE TEST
%
%  Generates N_test fresh noise realisations from the same Dryden model
%  used during training, then evaluates every controller in the bank on
%  those realisations.
%
%  No adversarial perturbation — pure i.i.d. test.
%
%  Outputs
%  -------
%    results  (table)  : one row per (rho, controller)
%    summary  (table)  : mean/std cost and violation rate per (rho, controller)
%
%  Saved to:  experiment_outputs_iid/
% =======================================================================

cfg.N_test   = 20000;   % number of fresh i.i.d. noise realisations
cfg.out_dir  = 'experiment_outputs_iid';
cfg.cache_path = fullfile('experiment_outputs_full', ...
                          'ac5_cached_controller_bank_N10_full.mat');

if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end

%% ------------------------------------------------------------------
%  System setup  (must match training)
% ------------------------------------------------------------------
[A, ~, B, ~, ~, ~, ~, ~, nx, ~, nu, ~, ~] = COMPleib('AC5');

sys.A  = A;
sys.B  = B;
sys.d  = nx;
sys.m  = nu;
sys.E  = eye(sys.d);
sys.Ts = 0.1;

opt.Qt = eye(sys.d);
opt.Rt = 0.01 * eye(sys.m);
opt.N  = 10;
opt.Q  = kron(eye(opt.N), opt.Qt);
opt.R  = kron(eye(opt.N), opt.Rt);
opt.C  = blkdiag(opt.Q, opt.R);

ic = [0.7; 0.1; 0.5; 0.3];   % fixed initial condition (same as training)

state_ub = [deg2rad(20); deg2rad(15); deg2rad(10); deg2rad(30)];
H_term   = [eye(sys.d); -eye(sys.d)];
b_term   = [-state_ub; -state_ub];

%% ------------------------------------------------------------------
%  Load controller bank
% ------------------------------------------------------------------
if exist('controller_bank', 'var') && ~isempty(controller_bank)
    controller_bank_local = controller_bank;
    fprintf('Using controller_bank from workspace (%d rho blocks).\n', ...
            numel(controller_bank_local));
else
    if ~isfile(cfg.cache_path)
        error('Cache not found at %s', cfg.cache_path);
    end
    S = load(cfg.cache_path, 'controller_bank');
    controller_bank_local = S.controller_bank;
    fprintf('Loaded controller_bank from %s\n', cfg.cache_path);
end

%% ------------------------------------------------------------------
%  Generate i.i.d. test realisations
%  Each column of W_test is one realisation: [ic; noise] flattened
% ------------------------------------------------------------------
w_dim = sys.d * opt.N;          % expected disturbance dimension (4*10=40)
W_test = zeros(w_dim, cfg.N_test);
fprintf('Generating %d i.i.d. noise realisations...', cfg.N_test);
for k = 1:cfg.N_test
    W_test(:, k) = [ic; process_noise(opt.N - 1, sys.Ts)];
end
fprintf(' done.\n');

%% ------------------------------------------------------------------
%  Evaluate every controller on the test set
% ------------------------------------------------------------------
% Collect rows in parallel arrays; build the table once at the end.
rows_rho      = [];
rows_ctrl     = string([]);
rows_eps      = [];
rows_cost     = [];
rows_std      = [];
rows_viol     = [];

for ir = 1:numel(controller_bank_local)
    rho_i         = controller_bank_local(ir).rho;
    controllers_i = controller_bank_local(ir).controllers;

    if isempty(controllers_i)
        warning('Empty controller block for rho=%.6g. Skipping.', rho_i);
        continue;
    end

    fprintf('\nEvaluating rho=%.6g (%d controllers)...\n', rho_i, numel(controllers_i));

    for c = 1:numel(controllers_i)
        ctrl = controllers_i(c);

        if size(ctrl.Phi_x, 2) ~= w_dim
            warning('Controller "%s" has wrong disturbance dim (%d, expected %d). Skipping.', ...
                    ctrl.name, size(ctrl.Phi_x, 2), w_dim);
            continue;
        end

        m = evaluate_controller(ctrl.Phi_x, ctrl.Phi_u, ctrl.phi_x, ctrl.phi_u, ...
                                W_test, opt, H_term, b_term, sys.d);

        rows_rho  (end+1) = rho_i;           %#ok<AGROW>
        rows_ctrl (end+1) = string(ctrl.name); %#ok<AGROW>
        rows_eps  (end+1) = ctrl.eps_train;  %#ok<AGROW>
        rows_cost (end+1) = m.mean_cost;     %#ok<AGROW>
        rows_std  (end+1) = m.std_cost;      %#ok<AGROW>
        rows_viol (end+1) = m.violation_rate; %#ok<AGROW>

        fprintf('  %-35s  mean_cost=%.4f  violation_rate=%.4f\n', ...
                ctrl.name, m.mean_cost, m.violation_rate);
    end
end

results = table(rows_rho(:), rows_ctrl(:), rows_eps(:), rows_cost(:), rows_std(:), rows_viol(:), ...
    'VariableNames', {'rho_train','controller','eps_train','mean_cost','std_cost','violation_rate'});

if isempty(results)
    error('No results produced — check controller_bank and dimensions.');
end

%% ------------------------------------------------------------------
%  Aggregate summary
% ------------------------------------------------------------------
summary = groupsummary(results, {'rho_train','controller'}, ...
                       {'mean','std'}, {'mean_cost','violation_rate'});

%% ------------------------------------------------------------------
%  Save
% ------------------------------------------------------------------
run_stamp    = datestr(now, 'yyyymmdd_HHMMSS');
results_path = fullfile(cfg.out_dir, sprintf('iid_oos_results_%s.csv', run_stamp));
summary_path = fullfile(cfg.out_dir, sprintf('iid_oos_summary_%s.csv', run_stamp));
noise_path   = fullfile(cfg.out_dir, sprintf('iid_test_noise_%s.mat',  run_stamp));

writetable(results, results_path);
writetable(summary, summary_path);
save(noise_path, 'W_test');

fprintf('\nDone.\n');
fprintf('Results : %s\n', results_path);
fprintf('Summary : %s\n', summary_path);
fprintf('Noise   : %s\n', noise_path);

%% =======================================================================
%  Local helper
% =======================================================================

function metrics = evaluate_controller(Phi_x, Phi_u, phi_x, phi_u, W, opt, H, b, d)
    % Vectorised over all n samples simultaneously.
    X_all = Phi_x * W + phi_x;           % (d*N) × n
    U_all = Phi_u * W + phi_u;           % (m*N) × n
    XU    = [X_all; U_all];              % (d*N + m*N) × n

    % Quadratic costs: diag(XU' * C * XU) without forming the full n×n product
    costs = sum(XU .* (opt.C * XU), 1)'; % n × 1

    % Terminal-state constraint violations
    XN    = X_all(end-d+1:end, :);       % d × n
    viols = any(H * XN + b > 0, 1)';    % n × 1 logical

    metrics.mean_cost      = mean(costs);
    metrics.std_cost       = std(costs);
    metrics.violation_rate = mean(viols);
end
