function W = process_noise(N, Ts, model)
% PROCESS_NOISE Creates process noise trajectories for B-747 model
%
%   W = process_noise(N, Ts)
%   W = process_noise(N, Ts, model)
%
%   Inputs:
%       N  - Number of time steps
%       Ts - Sampling time (e.g., 0.1s)
%       model - (optional) struct with Dryden model parameters. Any missing
%               fields are set to defaults.
%
%   Output:
%       W  - 4xN matrix of process noise flattened to a column vector. 
%            Original rows correspond to states: [beta; p; r; phi]
%
%   Physics (Dryden Continuous Turbulence Model):
%       - State 1 (Beta): Perturbed by lateral velocity noise (v_g)
%       - State 2 (p): Perturbed by roll rate noise (p_g)
%       - State 3 (r): Perturbed by yaw rate noise (r_g)
%       - State 4 (phi): Perturbed by integrated roll rate noise

    if nargin < 3 || isempty(model)
        model = struct();
    end

    model = fill_default_model(model);

    % --- 1. Aircraft and Environment Parameters ---
    V_tas = model.V_tas;      % True airspeed (ft/s)
    b = model.b;              % Wingspan (ft)
    
    L_v = model.L_v;          % Lateral scale length (ft)
    L_w = model.L_w;          % Vertical scale length (ft)
    
    sigma_v = model.sigma_v;  % Lateral turbulence intensity (ft/s)
    sigma_w = model.sigma_w;  % Vertical turbulence intensity (ft/s)
    
    % --- 2. Dryden Continuous Transfer Functions ---
    % Lateral Velocity Filter Hv(s)
    num_Hv = sigma_v * sqrt((2*L_v)/(pi*V_tas)) * [2*sqrt(3)*L_v/V_tas, 1];
    den_Hv = [4*(L_v/V_tas)^2, 4*L_v/V_tas, 1];
    H_v_cont = tf(num_Hv, den_Hv);
    
    % Roll Rate Filter Hp(s)
    num_Hp = sigma_w * sqrt(0.8/V_tas) * ((pi/(4*b))^(1/6)) / ((2*L_w)^(1/3));
    den_Hp = [4*b/(pi*V_tas), 1];
    H_p_cont = tf(num_Hp, den_Hp);
    
    % Yaw Rate Filter Hr(s)
    % Note: Hr(s) is a function of Hv(s)
    num_Hr_part = [-1/V_tas, 0];
    den_Hr_part = [3*b/(pi*V_tas), 1];
    H_r_cont = tf(num_Hr_part, den_Hr_part) * H_v_cont;
    
    % Discretize using Tustin to match simulation Ts
    H_v_discrete = c2d(H_v_cont, Ts, 'tustin');
    H_p_discrete = c2d(H_p_cont, Ts, 'tustin');
    H_r_discrete = c2d(H_r_cont, Ts, 'tustin');
    
    % --- 3. Generate Colored Wind Noise ---
    % Input to filter is band-limited Gaussian white noise
    % Scaled by 1/sqrt(Ts) for discrete-time power equivalence
    u_v = randn(N, 1) / sqrt(Ts);
    u_p = randn(N, 1) / sqrt(Ts);
    
    % Time vector for simulation
    t = (0:N-1)' * Ts;
    
    % lsim generates the filtered turbulence responses
    v_gust = lsim(H_v_discrete, u_v, t);
    p_gust = lsim(H_p_discrete, u_p, t);
    if model.shared_latent_input
        % Correlate lateral and yaw channels through shared Dryden input.
        u_r = u_v;
    else
        u_r = randn(N, 1) / sqrt(Ts);
    end
    r_gust = lsim(H_r_discrete, u_r, t);
    
    % --- 4. Map to State Dimensions ---
    w_beta = v_gust / V_tas;  % Convert lateral velocity to angle (radians)
    w_p = p_gust;             % Roll rate noise (rad/s)
    w_r = r_gust;             % Yaw rate noise (rad/s)
    w_phi = cumtrapz(t, w_p); % Roll angle noise is integral of roll rate noise
    
    % --- 5. Assemble Output Matrix W ---
    W = zeros(4, N);
    W(1, :) = w_beta'; 
    W(2, :) = w_p';    
    W(3, :) = w_r';    
    W(4, :) = w_phi';  
    
    % Flatten output to match original function design
    W = W(:);
end

function model = fill_default_model(model)
    model = set_if_missing(model, 'V_tas', 830);
    model = set_if_missing(model, 'b', 210);
    model = set_if_missing(model, 'L_u', 1750);
    model = set_if_missing(model, 'L_v', model.L_u / 2);
    model = set_if_missing(model, 'L_w', model.L_u / 2);
    model = set_if_missing(model, 'sigma_v', 20);
    model = set_if_missing(model, 'sigma_w', 20);
    model = set_if_missing(model, 'shared_latent_input', true);
end

function s = set_if_missing(s, field_name, default_value)
    if ~isfield(s, field_name) || isempty(s.(field_name))
        s.(field_name) = default_value;
    end
end