function [theta, alpha] = directQCQP(Q, Z, rho)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This function computes the optimization problem of Theorem 12 of the
    % paper Wasserstein DRO from theory to machine learning.
    % Given a cost function Q, radius rho and sample points Z, the function
    % returns the displacement of the worst-case distribution support
    % points from the nominal ones in theta
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    N = size(Z, 2);
    H = kron(eye(N), Q) ./ N;
    xi_hat = reshape(Z, [], 1);
    L = max(eig(Q));

    %% Direct solution of the QCQP with Gurobi
    theta = sdpvar(2*N, 1);
    alpha = sdpvar();
    Constraints = [];
    Constraints = [Constraints, theta' * H * theta + alpha <= rho, alpha >=0];

    Objective = (xi_hat + theta)' * H * (xi_hat + theta) + alpha * L;

    % Solve the optimization problem
    fprintf('=====================================\n')
    fprintf("Solving the optimization problem...\n")
    fprintf('=====================================\n')
    options = sdpsettings('verbose', 2, 'solver', 'gurobi');

    diagnostics = optimize(Constraints, -Objective, options);

    if diagnostics.problem ~= 0
        disp('Solver failed:');
        disp(diagnostics.info);
    end

    % obj = value(Objective);
    theta = value(theta);
    alpha = value(alpha);
end