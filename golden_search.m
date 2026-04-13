function [cost, lambda_opt, Phi_x, Phi_u, phi_x, phi_u] = golden_search(rho, eps, tol, lam_min, lam_max, sys, sls, opt, const, eps_c, rho_c)
    % Taken from https://drlvk.github.io/nm/section-golden-section.html
    % For a reference https://en.wikipedia.org/wiki/Golden-section_search#Iterative_algorithm

    gr = .5*(3-sqrt(5));
    a = lam_min; b = lam_max;
    interval = b - a;
    c = a + gr*interval;
    d = b - gr*interval;
    [~, ~, ~, ~, fc] = helperFunction(sys, sls, opt, c, rho, eps, const, eps_c, rho_c);
    [~, ~, ~, ~, fd] = helperFunction(sys, sls, opt, d, rho, eps, const, eps_c, rho_c);
    optimal = true;
    while abs(b-a) >= tol
        if fc < fd
            optimal = true;
            b = d;
            d = c;
            fd = fc;
            c = a + gr*(b-a);
            [Phi_x, Phi_u, phi_x, phi_u, fc] = helperFunction(sys, sls, opt, c, rho, eps, const, eps_c, rho_c);
        else
            optimal = false;
            a = c;
            c = d;
            fc = fd;
            d = b - gr*(b-a);
            [Phi_x, Phi_u, phi_x, phi_u, fd] = helperFunction(sys, sls, opt, d, rho, eps, const, eps_c, rho_c);
            
        end
    end
    if optimal
        cost = fc;
        lambda_opt = c;
    else
        cost = fd;
        lambda_opt = d;
    end
end

function [Phi_x, Phi_u, phi_x, phi_u, objective] = helperFunction(sys, sls, opt, lam, rho, eps, const, eps_c, rho_c)
    
    % Define the decision variables of the optimization problem
    m = sys.d + sys.p*(opt.N-1);
    Phi_u = sdpvar(sys.m*opt.N, m, 'full');
    Phi_x = (sls.I - sls.Z*sls.A) \ (sls.Z*sls.B*Phi_u + sls.E); % Phi_x is given as function of Phi_u
    Phi = [Phi_x; Phi_u];

    phi_u = sdpvar(sys.m*opt.N, 1);
    phi_x = (sls.I - sls.Z*sls.A) \ (sls.Z*sls.B*phi_u);
    phi = [phi_x; phi_u];

    s = sdpvar(opt.n, 1);   % epigraphical variable
    z = sdpvar(opt.n, 1);   % auxiliary variable for logdet part
    Q = sdpvar(m, m, 'symmetric');
    q = sdpvar(m ,1);
    c = sdpvar();   
    
    % define the objective function
    objective = lam*rho + sum(s)/opt.n;

    tau = sdpvar();   % tau variable of CVaR
    sigma = sdpvar(); % dual variable of the Sinkhorn problem
    zeta = sdpvar(opt.n,1);  % epigraphical variable for the CVaR
    H = [const.H; zeros(1, sys.d)];
    b = [const.b; tau];
    I = eye(size(opt.Sigma));
    
    % Define the constraints
    constraints = [sigma >= 0, sigma*rho_c + (const.gamma-1)/const.gamma*tau + sum(zeta)/opt.n <= 0];
    
    % second LMI constraint
    D_half = sqrtm(opt.C);
    LMI2 = [[Q q; q' c], [Phi phi]'*D_half'; D_half*[Phi phi], eye((sys.m+sys.d)*opt.N)];
    constraints = [constraints, LMI2 >= 0];

    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.N-2
        for j = i+1:opt.N-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.p):((j+1)*sys.p)) == zeros(sys.m, sys.p)];
        end
    end

    matr = lam .* (eye(m) + .5 * eps .* inv(opt.Sigma)) - Q;

    nonlin = eps * lam * .5 * m * (log(lam * eps * .5) - log(geomean(matr))) - eps * lam * .5 * logdet(opt.Sigma);

    for i=1:opt.n
        % Get the i-th datapoint
        wi_hat = opt.process_noise(:, i);
        % First inequality constraint
        constraints = [constraints, s(i) >= z(i) + nonlin];

        % First LMI constraint
        tmp = q + lam * (wi_hat + eps * .5 .* opt.Sigma \ opt.m);
        LMI = [matr, tmp; tmp', z(i) - c + lam * (wi_hat'*wi_hat) + eps * lam * .5 .* opt.m' * (opt.Sigma \ opt.m)];
        constraints = [constraints, LMI >= 0];

        for j = 1:size(H, 1)

            seg = (H(j, :) * Phi_x(end-sys.d+1:end, :) ./ const.gamma)' + 2*sigma*wi_hat + sigma*eps_c*(opt.Sigma \ opt.m);

            s_i_term = zeta(i) - b(j)/const.gamma - (H(j, :) * phi_x(end-sys.d+1:end))/const.gamma + sigma*(wi_hat'*wi_hat) - sigma*eps_c*m*0.5*log(eps_c)...
                       + sigma*eps_c*0.5*logdet(2*opt.Sigma+eps_c*I) + 0.5*sigma*eps_c*(opt.m'* (opt.Sigma \ opt.m));

            L = 4*sigma*eye(size(wi_hat, 1)) + sigma*eps_c*inv(opt.Sigma);

            M = [L seg; seg' s_i_term];

            constraints = [constraints, M >= 0];
        end
    end
    
    % Solve the optimization problem
    fprintf('=====================================')
    fprintf("Solving the optimization problem...")
    fprintf('=====================================\n')
    options = sdpsettings('verbose', 0, 'solver', 'mosek');

    sol = optimize(constraints, objective, options);
    
    % Q = value(Q);
    % if lam * (eye(m) + .5*eps./opt.Sigma) <= Q
    %     objective = inf;
    % end

    if ~(sol.problem == 0)
        if sol.problem == 1
            objective = inf;
            return
        elseif sol.problem == 4
            objective = value(objective);
        else
            string = yalmiperror(sol.problem);
            error(string);
        end
    end

    fprintf('Value of lambda: %s Result: %s\n', num2str(lam), num2str(value(objective)));
    
    % Extract the closed-loop responses corresponding to a unconstrained causal 
    % linear controller that is optimal
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    phi_x = value(phi_x);
    phi_u = value(phi_u);

    % Extract the cost incurred by an unconstrained causal linear controller
    objective = value(objective);
end