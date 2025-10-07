close all; clearvars; clc;
rng(23);             % Set random seed for reproducibility

%% Definition of the underlying discrete-time LTI system
rho = .1; % Wasserstein radius

% System dynamics
sys.A = [.9 1; 0 0.9];
sys.B = [1;1];

sys.d = size(sys.A, 1);   % Order of the system: state dimension
sys.m = size(sys.B, 2);   % Number of input channels
sys.p = sys.d;
sys.E = eye(sys.d);

% Definition of the parameters of the optimization problem
opt.Qt = 1000*eye(sys.d); % Stage cost: state weight matrix
opt.Rt = eye(sys.m); % Stage cost: input weight matrix

opt.N = 5; % Control horizon

opt.Q = kron(eye(opt.N), opt.Qt); % State cost matrix
opt.R = kron(eye(opt.N), opt.Rt); % Input cost matrix
opt.C = blkdiag(opt.Q, opt.R); % Cost matrix

opt.m = zeros(sys.d*opt.N, 1); 
opt.Sigma_t = .1*eye(sys.d);
opt.Sigma = kron(eye(opt.N), opt.Sigma_t);

%% Definition of the stacked system dynamics over the control horizon
sls.A = kron(eye(opt.N), sys.A);
sls.B = kron(eye(opt.N), sys.B);
sls.E = blkdiag(eye(sys.d), kron(eye(opt.N-1), sys.E));

% Identity matrix and block-downshift operator
sls.I = eye(sys.d*opt.N);
sls.Z = [zeros(sys.d, sys.d*(opt.N-1)) zeros(sys.d, sys.d); eye(sys.d*(opt.N-1)) zeros(sys.d*(opt.N-1), sys.d)];

%% Generation of noise samples

opt.n = 5; % Number of noise datapoints
mean_vector = zeros(sys.d, 1); % Zero mean

% Generate random noise datapoints
data_points = cell(opt.n, 1); % Initialize an empty cell array
for i = 1:opt.n
    % Gaussian samples
    trajectory = mvnrnd(mean_vector, opt.Sigma_t, opt.N); % Size: N x d
    data_points{i} = reshape(trajectory', [], 1);
end
opt.data = data_points;

%% Definition of the constraints
H = [-1 0; 1 0; 0 -1; 0 1; 0 0];
l1 = -5;
l2 = 1;
u1 = -3;
u2 = 2;

b = [l1; -u1; l2; -u2];

rho_c = .01;
gamma = .2;

%% Wasserstein DRControl

% Define the decision variables of the optimization problem
m = sys.d + sys.p*(opt.N-1);
Phi_u = sdpvar(sys.m*opt.N, m, 'full');
Phi_x = (sls.I - sls.Z*sls.A) \ (sls.Z*sls.B*Phi_u + sls.E); % Phi_x is given as function of Phi_u
Phi = [Phi_x; Phi_u];

phi_u = sdpvar(sys.m*opt.N, 1);
phi_x = (sls.I - sls.Z*sls.A) \ (sls.Z*sls.B*phi_u);
phi = [phi_x; phi_u];

s = sdpvar(opt.n, 1);
lambda = sdpvar();
Q = sdpvar(m, m, 'symmetric');
q = sdpvar(m ,1);
c = sdpvar();
tau = sdpvar();
sigma = sdpvar();
z = sdpvar(opt.n, 1);
b = [b;tau];

% define the objective function
objective = lambda*rho + sum(s)/opt.n;

% define the constraints
constraints = [sigma >= 0, sigma*rho_c + (gamma - 1)/gamma*tau + sum(z)/opt.n <= 0];

% second LMI constraint
D_half = sqrtm(opt.C);
LMI2 = [[Q q; q' c], [Phi phi]'*D_half'; D_half*[Phi phi], eye((sys.m+sys.d)*opt.N)];
% LMI2 = [Q , Phi'*D_half'; D_half*Phi, eye((sys.m+sys.d)*opt.N)];
constraints = [constraints, LMI2 >= 0];

% Impose the causal sparsities on the closed loop responses
for i = 0:opt.N-2
    for j = i+1:opt.N-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
        constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.p):((j+1)*sys.p)) == zeros(sys.m, sys.p)];
    end
end

% Loop for each datapoint 
for i=1:opt.n
    % Get the i-th datapoint
    xi_hat = opt.data{i};

    % First LMI constraint
    LMI = [lambda*eye(m)-Q, q + lambda*xi_hat; q' + lambda*xi_hat', s(i) - c + lambda*(xi_hat'*xi_hat)];
    % LMI = [lambda*eye(m)-Q, lambda*xi_hat; lambda*xi_hat', s(i) + lambda*(xi_hat'*xi_hat)];
    constraints = [constraints, LMI >= 0];

    for j=1:size(H, 1)
        
        z_i = z(i) - b(j)/gamma - (H(j, :) * phi_x(end-sys.d+1:end))/gamma + sigma * (xi_hat'*xi_hat);
        diag = (H(j, :) * Phi_x(end-sys.d+1:end, :))'./gamma + 2*sigma*xi_hat;
        L = 4*sigma*eye(size(xi_hat, 1));
        LMIc = [L diag; diag' z_i];
        constraints = [constraints, LMIc >= 0];
    end
end

% Solve the optimization problem
fprintf('=====================================')
fprintf("Solving the optimization problem...")
fprintf('=====================================')
options = sdpsettings('verbose', 2, 'solver', 'mosek');
sol = optimize(constraints, objective, options);
if ~(sol.problem == 0)
    disp(sol.problem)
    error('Something went wrong...');
end

% Extract the closed-loop responses corresponding to a unconstrained causal 
% linear controller that is optimal
Phi_x = value(Phi_x); 
Phi_u = value(Phi_u);
phi_x = value(phi_x);


%% Plots
colors = {[1,0,0], [0,0,1], [0,0.5,0], [0.5,0,0.5]}; % red, blue, green, purple
figure; hold on;
% legend(labels);
% Define constraint set for visualization
rectangle('Position',[l1, l2, u1-l1, u2-l2], 'FaceColor',[0.85 0.85 0.85], 'EdgeColor','none');

% colors = lines(length(rho_c));

% Loop over different epsilons
for k=1:opt.n
    % Use different w_samples if needed (here assumed same)
    x_samples = Phi_x * data_points{k} + phi_x;
    x_samples = reshape(x_samples, sys.d, opt.N);
    plot(x_samples(1, :), x_samples(2, :), '-o', 'LineWidth', 0.6, 'MarkerSize', 5);
    hold on;
    % Plot the last point as a cross
    plot(x_samples(1, end), x_samples(2, end), 'x','MarkerSize', 8, 'LineWidth', 1.2);
end
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
axis equal;
grid on;
