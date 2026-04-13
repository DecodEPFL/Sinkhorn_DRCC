function [lambda_opt, Phi_x, Phi_u, phi_x, phi_u, cost] = train_wasserstein_controller(sys, sls, opt, const, rho, rho_c)
    H = [const.H; zeros(1, sys.d)];
    b = const.b;
    gamma = const.gamma;

    m = sys.d * opt.N;

    Phi_u = sdpvar(sys.m * opt.N, m, 'full');
    Phi_x = (sls.I - sls.Z * sls.A) \ (sls.Z * sls.B * Phi_u + sls.I);
    Phi = [Phi_x; Phi_u];

    phi_u = sdpvar(sys.m * opt.N, 1);
    phi_x = (sls.I - sls.Z * sls.A) \ (sls.Z * sls.B * phi_u);
    phi = [phi_x; phi_u];

    s = sdpvar(opt.n, 1);
    lambda = sdpvar();
    Q = sdpvar(m, m, 'symmetric');
    q = sdpvar(m, 1);
    c = sdpvar();

    tau = sdpvar();
    sigma = sdpvar();
    z = sdpvar(opt.n, 1);
    b = [b; tau];

    objective = lambda * rho + sum(s) / opt.n;

    constraints = [sigma >= 0, sigma * rho_c + (gamma - 1) / gamma * tau + sum(z) / opt.n <= 0];

    D_half = sqrtm(opt.C);
    LMI2 = [[Q q; q' c], [Phi phi]' * D_half'; D_half * [Phi phi], eye((sys.m + sys.d) * opt.N)];
    constraints = [constraints, LMI2 >= 0];

    for i = 0:opt.N-2
        for j = i+1:opt.N-1
            rows_u = (1 + i * sys.m):((i + 1) * sys.m);
            cols_w = (1 + j * sys.d):((j + 1) * sys.d);
            constraints = [constraints, Phi_u(rows_u, cols_w) == zeros(sys.m, sys.d)];
        end
    end

    for i = 1:opt.n
        xi_hat = opt.process_noise(:, i);

        LMI = [lambda * eye(m) - Q, q + lambda * xi_hat; ...
               q' + lambda * xi_hat', s(i) - c + lambda * (xi_hat' * xi_hat)];
        constraints = [constraints, LMI >= 0];

        for j = 1:size(H, 1)
            z_i = z(i) - b(j) / gamma - (H(j, :) * phi_x(end - sys.d + 1:end)) / gamma + sigma * (xi_hat' * xi_hat);
            diag_term = (H(j, :) * Phi_x(end - sys.d + 1:end, :))' ./ gamma + 2 * sigma * xi_hat;
            L = 4 * sigma * eye(size(xi_hat, 1));
            LMIc = [L diag_term; diag_term' z_i];
            constraints = [constraints, LMIc >= 0];
        end
    end

    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);

    if sol.problem ~= 0
        error('Wasserstein optimization failed: %s', yalmiperror(sol.problem));
    end

    lambda_opt = value(lambda);
    Phi_x = value(Phi_x);
    Phi_u = value(Phi_u);
    phi_x = value(phi_x);
    phi_u = value(phi_u);
    cost = value(objective);
end