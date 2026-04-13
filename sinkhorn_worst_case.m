close all; clearvars; clc;
rng(0);
%%
epsilons = [5e-3, 0.05, .5];
rho     = 1;
Q       = eye(2);
lam_min = 0; lam_max = 100;
data = [.25, .75; .75, .25]; % 2 samples, 2D
% Golden search
gr = .5*(3-sqrt(5));
optimal = true;
tol = 1e-3;
grid_size = 100;
z1 = linspace(0, 2, grid_size);
z2 = linspace(0, 2, grid_size);
[Z1, Z2] = meshgrid(z1, z2);
z_vals = [Z1(:), Z2(:)];

theta = directQCQP(Q, data, rho);

%% Computations
figure;
t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
nexttile;
h1 = plot(data(:,1), data(:,2), 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Support points (nominal distribution)');
hold on;
h2 = plot(data(:, 1) + theta(1:2) , data(:, 2) + theta(3:4), 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Support points (worst-case Wasserstein distribution)');
% legend('show', 'Orientation','vertical', ...
%     'Box','off');
% legend([h1, h2], {'Support points (nominal distribution)', 'Support points (worst-case distribution)'}, 'Location', 'best');
grid on;
% Draw arrow using quiver
quiver(data(1, 1), data(2, 1), theta(1), theta(2), 0, 'MaxHeadSize', 0.3, 'Color', 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
quiver(data(1, 2), data(2, 2), theta(3), theta(4), 0, 'MaxHeadSize', 0.3, 'Color', 'k', 'LineWidth', 1, 'HandleVisibility', 'off');

axis([0 2 0 2]);
set(gca, 'Color', [0.95 0.95 0.95])
title('Worst-case transport mapping for W-DRO', 'Interpreter', 'latex');
set(gcf, 'Color', [1 1 1])
for i=1:size(epsilons, 2)
    nexttile;
    epsilon = epsilons(i);
    a = lam_min; b = lam_max;
    interval = b - a;
    c = a + gr*interval;
    d = b - gr*interval;
    fc = helperFunction(c, Q, rho, epsilon, data);
    fd = helperFunction(d, Q, rho, epsilon, data);
    while abs(b-a) >= tol
        if fc < fd
            optimal = true;
            b = d;
            d = c;
            fd = fc;
            c = a + gr*(b-a);
            fc = helperFunction(c, Q, rho, epsilon, data);
        else
            optimal = false;
            a = c;
            c = d;
            fc = fd;
            d = b - gr*(b-a);
            fd = helperFunction(d, Q, rho, epsilon, data);
            
        end
    end
    
    if optimal
        cost = fc;
        lambda_opt = c;
    else
        cost = fd;
        lambda_opt = d;
    end
    
    %% Visualization
    % Generate samples from N(0, I)
    density_vals = zeros(size(z_vals,1), 1);

    % Compute density values using your density() function
    for j = 1:size(z_vals, 1)
        density_vals(j, :) = density(z_vals(j,:), Q, lambda_opt, epsilon, data);
    end
    
    density_vals = reshape(density_vals, grid_size, grid_size);
    
    % Plot
    % legend(data, {'Support points (nominal distribution)'}, 'Location', 'best');
    contourf(Z1, Z2, density_vals, 8, 'LineColor', 'none', 'HandleVisibility', 'off');
    % colorbar;
    hold on;
    plot(data(:,1), data(:,2), 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Support (nominal distribution)');
    % legend('show', 'Location', 'best');
    % plot(data(:, 1) + theta(1:2) , data(:, 2) + theta(3:4), 'bo', 'MarkerFaceColor', 'b')
    % xlabel('$z_1$', 'Interpreter', 'latex');
    % ylabel('$z_2$', 'Interpreter', 'latex');
    title(['Worst-case distribution for S-DRO $(\epsilon =\, $' num2str(epsilon) '$)$'], 'Interpreter', 'latex');
    colormap('sky');  
end
% Create a common legend attached to the tiledlayout
% lgd = legend(t, [h1 h2], ...
%     'Orientation','vertical', ...
%     'Location','southoutside', ...
%     'Box','off');
% lgd.FontSize = 9;
leg = legend([h1 h2], 'Orientation', 'horizontal');
leg.Layout.Tile = 'south';

set(gcf, 'Units', 'centimeters');
afFigurePosition = [1 1 20 14]; % [pos_x pos_y width_x width_y]
set(gcf, 'Position', afFigurePosition); % [left bottom width height]
set(gcf, 'PaperPositionMode', 'auto');


set(gca, 'Units','normalized',... %
'Position',[0.15 0.2 0.75 0.7]);
exportgraphics(gcf, 'Sinkhorn_worst_case.pdf', 'ContentType', 'vector');

%% Functions (Gaussian reference)
function cost = helperFunction(lambda, Q, rho, epsilon, data)
    % This function computes the optimal lambda for the Sinkhorn dual

    S = lambda * (1 + 0.5 * epsilon)*eye(2) - Q;

    if min(eig(S)) <= 0
        cost = inf;
        return
    end

    cost = lambda*rho;
    % Constant term
    term1 = (lambda * epsilon) * log(lambda * epsilon / 2);
    term2 = - (lambda * epsilon / 2);
    term3 = - (lambda * epsilon / 2) * log(det(S));

    cost = cost + term1 + term2 + term3;
    sum = 0;
    for i=1:size(data, 1)
        xi = data(i, :)';

        tmp = lambda * xi;
        quad_term = tmp' * (S \ tmp);

        sum = sum + quad_term - lambda * norm(xi)^2;

    end
    cost = cost + sum/size(data,1);
end

function dens = density(z_val, Q, lam, eps, x_samples)
    % This function computes the density of the Sinkhorn worst-case
    % distribution on a set of points z_val given the training dataset
    % x_samples.

    dim = size(z_val,2);

    % Compute alpha(x) for each sample 
    alpha_vals = alpha_x(x_samples, lam, eps, Q);  % N×1 vector

    % Compute cost: ||x - z||^2 for each row
    diffs = x_samples - z_val;
    cost = sum(diffs.^2, 2); 

    quad = z_val * Q * z_val';

    % Compute exp_term for all samples
    exp_term = exp((quad - lam * cost) / (lam * eps)); 

    % Compute the expectation value for this z
    expectation_term = mean(exp_term ./ alpha_vals);

    % Standard Gaussian PDF evaluated at z_vals
    gaussian_pdf = mvnpdf(z_val, zeros(1, dim), eye(dim)); 

    % Final density
    dens = gaussian_pdf * expectation_term; 
end

function values = alpha_x(xi, lambda_opt, epsilon, Q)
    % This function computes the normalization factor \alpha_x for each
    % datapoint \xi

    d = size(xi, 2);

    S = lambda_opt * (1 + epsilon * .5) * eye(d) - Q;
    alpha = lambda_opt * epsilon / 2 / sqrt(det(S));

    % Vectorized quadratic form: row-wise xi * inv(S) * xi'
    quad_terms = sum(xi * S \ xi, 2);
    norms = sum(xi.^2, 2);
    tmp = (lambda_opt^2 * quad_terms - lambda_opt * norms) / (lambda_opt * epsilon);

    values = 1 ./ (alpha * exp(tmp));

end

% %% Functions (Lebesgue reference)
% 
% function cost = helperFunction(lambda, Q, rho, epsilon, data)
% 
%     S = lambda * eye(2) - Q;
% 
%     if min(eig(S)) <= 0
%         cost = inf;
%         return
%     end
% 
%     cost = lambda*rho;
%     % Constant term
%     term1 = lambda * epsilon * log(lambda * epsilon * pi);
%     % term2 = - (lambda * epsilon / 2);
%     term3 = - (lambda * epsilon / 2) * log(det(S));
% 
%     cost = cost + term1 + term3;
%     sum = 0;
%     for i=1:size(data, 1)
%         xi = data(i, :)';
%         tmp = lambda * xi;
%         quad_term = tmp' * (S \ tmp);
%         sum = sum + quad_term - lambda * norm(xi)^2;
% 
%     end
%     cost = cost + sum/size(data,1);
% end
% 
% function dens = density(z_val, Q, lam, eps, x_samples)
% 
%     % dim = size(z_val,2);
% 
%     % Compute alpha(x) for each sample 
%     alpha_vals = alpha_x(x_samples, lam, eps, Q);  % N×1 vector
% 
%     % Compute cost: ||x - z||^2 for each row
%     diffs = x_samples - z_val;
%     cost = sum(diffs.^2, 2); 
% 
%     quad = z_val * Q * z_val';
% 
%     % Compute exp_term for all samples
%     exp_term = exp((quad - lam * cost) / (lam * eps)); 
% 
%     % Compute the expectation value for this z
%     expectation_term = mean(exp_term ./ alpha_vals);
% 
%     % Standard Gaussian PDF evaluated at z_vals
%     % gaussian_pdf = mvnpdf(z_val, zeros(1, dim), eye(dim)); 
% 
%     % Final density
%     dens = expectation_term; 
% end
% 
% function values = alpha_x(xi, lambda_opt, epsilon, Q)
% 
%     d = size(xi, 2);
% 
%     S = lambda_opt * eye(d) - Q;
%     alpha = lambda_opt * epsilon * pi / sqrt(det(S));
% 
%     % Vectorized quadratic form: row-wise xi * inv(S) * xi'
%     quad_terms = sum(xi * S \ xi, 2);
%     norms = sum(xi.^2, 2);
%     tmp = (lambda_opt * quad_terms - norms) / epsilon;
% 
%     values = 1 ./ (alpha * exp(tmp));
% 
% end