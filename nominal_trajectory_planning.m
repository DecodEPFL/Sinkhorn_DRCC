function [Phi_x, Phi_u, phi_x, phi_u, ret] = nominal_trajectory_planning(sys, sls, opt, const)
    % Compute the trajectory planning linear controller using the empirical
    % center distribution.
    % This implementation uses the Wasserstein CVaR constraints with ε=0, I
    % was lazy to derive the correct expression with empirical distribution.

    H = [const.H; zeros(1, sys.d)];
    b = const.b;
    gamma = const.gamma;

    % Define the decision variables of the optimization problem
    m = sys.d + sys.p*(opt.N-1);
    Phi_u = sdpvar(sys.m*opt.N, m, 'full');
    Phi_x = (sls.I - sls.Z*sls.A) \ (sls.Z*sls.B*Phi_u + sls.I); % Phi_x is given as function of Phi_u
    Phi = [Phi_x; Phi_u];

    phi_u = sdpvar(sys.m*opt.N, 1);
    phi_x = (sls.I - sls.Z*sls.A) \ (sls.Z*sls.B*phi_u);
    phi = [phi_x; phi_u];

    tau = sdpvar();
    sigma = sdpvar();
    z = sdpvar(opt.n, 1);
    b = [b; tau];

    % define the objective function
    objective = 0;

    % define the constraints
    constraints = [sigma >= 0, (gamma - 1)/gamma*tau + sum(z)/opt.n <= 0];

    for i=1:opt.n

        % Get the i-th datapoint
        xi_hat = opt.process_noise(:, i);
        tmp = Phi*xi_hat;
        objective = objective + tmp'*opt.C*tmp;
    
        for j=1:size(H, 1)
        
            z_i = z(i) - b(j)/gamma - (H(j, :) * phi_x(end-sys.d+1:end))/gamma + sigma * (xi_hat'*xi_hat);
            diag = (H(j, :) * Phi_x(end-sys.d+1:end, :))'./gamma + 2*sigma*xi_hat;
            L = 4*sigma*eye(size(xi_hat, 1));
            LMIc = [L diag; diag' z_i];
            constraints = [constraints, LMIc >= 0];

        end

    end

    objective = (objective + phi' * opt.C * phi) / opt.n;

    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.N-2
        for j = i+1:opt.N-1
            rows_u = (1 + i * sys.m):((i + 1) * sys.m);
            cols_w = (1 + j * sys.d):((j + 1) * sys.d);
            constraints = [constraints, Phi_u(rows_u, cols_w) == zeros(sys.m, sys.d)];
        end
    end

    % Explicitly set the completely unconstrained final input block to zero
    % THIS SHOULDN'T BE NECESSARY BUT OTHERWISE MOSEK RETURNS SOME NaN
    % VALUES
    constraints = [constraints, Phi_u(end-sys.m+1:end, :) == 0];

    % Solve the optimization problem
    fprintf('=====================================')
    fprintf("Solving the optimization problem...")
    fprintf('=====================================')
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    % Extract the closed-loop responses corresponding to a unconstrained causal 
    % linear controller that is optimal
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    phi_x = value(phi_x);
    phi_u = value(phi_u);
    
    % Extract the cost incurred by an unconstrained causal linear controller
    ret = value(objective);
end