%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computational Assignment – Summary & Methods
% -------------------------------------------------------------------------
% Author: Heng Tang
% -------------------------------------------------------------------------
% Goal:
%   Numerical solutions and comparisons for two problems:
%   dynamic resource extraction and a real-options drilling problem.
% -------------------------------------------------------------------------
% Task 1 – Discrete DDP with exact hits (deterministic transition)
%   Set up a discrete state/action grid (stocks 0–1000, step 2; actions = states),
%   build flow utility and deterministic transition blocks, solve by value
%   function iteration, and simulate extraction/stock paths and implied prices.
% -------------------------------------------------------------------------
% Task 2 – Interpolation between states/actions
%   Keep the state grid; make the action grid denser near zero by spacing
%   evenly in sqrt-space; construct an interpolated transition matrix and
%   use policy interpolation in simulation to obtain smoother paths.
% -------------------------------------------------------------------------
% Task 3 – Use the FOC (analytical policy paths)
%   From the FOC δ^t u'(x_t) = μ0, derive closed-form production paths for
%   each utility; choose μ0 so cumulative production matches S_tot; plot the
%   smooth time paths implied by the FOC.
% -------------------------------------------------------------------------
% Task 4 – Drilling and real options (optimal stopping)
%   Price follows P_{t+1}=P_t+ε, ε~N(0,4^2). Build a price grid and a
%   transition matrix via midpoint bins and normcdf. Solve the stopping
%   problem V(P)=max{PX–D, δ·E[V(P')|P]} by value iteration, report the
%   trigger price P* (~$41), and plot the option value V(P).
% -------------------------------------------------------------------------
% Outputs:
%   - Tasks 1/2: Optimal policies, state transitions, (X_t, S_t) paths, and
%                price paths.
%   - Task 3   : FOC-implied smooth X_t paths (calibrated via μ0).
%   - Task 4   : Transition matrix T, option value V(P), trigger price P*,
%                and the value-function plot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% -------------General setup for task 1,2,3-------------
% Inital resouce stock: S_tot = 1000
% Extraction cost: c = 0
% Discount rate: r = 0.05, thus discount factor: delta = 1/(1+r)
% Time horizon: T = infinite periods
% Utility function 1: u(y) = 2 * y^(1/2)
% Utility function 2: u(y) = 5y - 0.05y^2

% -------------Task 1: Solving the DDP by value function iteration; discrete state space.-------------
% For both utility functions, solve for the optimal extraction path on the computer, following the steps below:

% a) Set up the state space (stock remaining) so that it runs from 0 to 1000 in 501 steps (so that the step size is 2). 
% Let S denote the state space vector, and let N = 501 denote the number of discrete states.
clc;
clear;
S_tot = 1000;
N = 501;
S = linspace(0, S_tot, N)'; % State space vector
delta = 1/(1 + 0.05); % Discount factor

% b) Set up the action space (the amount to extract each period) so that it matches the state space. 
A = S; % Action space vector
nA = N; % Number of possible actions

% c) Define utility given extraction.
u1 = @(x) 2 * x .^ 0.5; % Utility function 1
u2 = @(x) 5 * x - 0.05 * x .^ 2; % Utility function 2

% d) Define a flow utility matrix U (N x nA) that gives the flow utility for every state and action.
% let U = -∞ for all cells in which the action implies more extraction than the remaining stock.
U1 = -inf(N, nA); % Flow utility matrix for utility function 1
U2 = -inf(N, nA); % Flow utility matrix for utility function 2
for i = 1:N
    for j = 1:nA
        if A(j) <= S(i)
            U1(i, j) = u1(A(j));
            U2(i, j) = u2(A(j));
        end
    end
end

% e) Create another N x nA matrix that identifies the index of the state next period, given the current state and action.
next_state_index = zeros(N, nA); % Next state index matrix
for i = 1:N
    for j = 1:nA
        remaining_stock = S(i) - A(j);
        if remaining_stock < 0
            next_state_index(i, j) = 1; % Zero stock state
        else
            [~, idx] = min(abs(S - remaining_stock));
            next_state_index(i, j) = idx;
        end
    end
end

% f) Create a state transition matrix T that is N x (N*nA). 
T = sparse(N, N * nA); % State transition matrix
for j = 1:nA
    for i = 1:N
        next_idx = next_state_index(i, j);
        T(i, (j - 1) * N + next_idx) = 1;
    end
end

% g) Initialize the value and control functions, and solve the model using value function iteration.
tolerance = 1e-8;
max_iterations = 10000;
V1 = zeros(N, 1); % Value function for utility function 1
V2 = zeros(N, 1); % Value function for utility function 2
C1 = zeros(N, 1); % Control function for utility function 1
C2 = zeros(N, 1); % Control function for utility function 2
for iter = 1:max_iterations
    Vnext1 = zeros(N, nA);
    Vnext2 = zeros(N, nA);
    for j = 1:nA
        Vnext1(:, j) = T(:, (j - 1) * N + 1:j * N) * V1;
        Vnext2(:, j) = T(:, (j - 1) * N + 1:j * N) * V2;
    end
    Vnew1 = U1 + delta * Vnext1;
    Vnew2 = U2 + delta * Vnext2;
    [V1_new, C1_new] = max(Vnew1, [], 2);
    [V2_new, C2_new] = max(Vnew2, [], 2);
    if max(abs(V1_new - V1)) < tolerance && max(abs(V2_new - V2)) < tolerance
        V1 = V1_new; C1 = C1_new;
        V2 = V2_new; C2 = C2_new;
        break;
    end
    V1 = V1_new; C1 = C1_new;
    V2 = V2_new; C2 = C2_new;
end

X1_opt = A(C1);
X2_opt = A(C2);

% h) Find the N by N optimal transition matrix Topt for each utility function
Topt1 = sparse(N, N); % Optimal transition matrix for utility function 1
Topt2 = sparse(N, N); % Optimal transition matrix for utility function 2
for i = 1:N
    optimal_action_idx1 = C1(i);
    optimal_action_idx2 = C2(i);
    next_idx1 = next_state_index(i, optimal_action_idx1);
    next_idx2 = next_state_index(i, optimal_action_idx2);
    Topt1(i, next_idx1) = 1;
    Topt2(i, next_idx2) = 1;
end

% i) Simulate the model for t = 80 periods, starting with the initial stock of 1000. 
T_sim = 80;
idx_path1 = zeros(T_sim,1);
idx_path2 = zeros(T_sim,1);
S_path1   = zeros(T_sim,1);
S_path2   = zeros(T_sim,1);
X_path1   = zeros(T_sim,1);
X_path2   = zeros(T_sim,1);

[~, idx0] = min(abs(S - S_tot));
idx_path1(1) = idx0;
idx_path2(1) = idx0;
S_path1(1)   = S(idx0);
S_path2(1)   = S(idx0);
% use the optimal C and Topt mappings to obtain each period’s extraction and the stock remaining at the end of the period.
for t = 1:T_sim-1
    % --- Utility function 1 ---
    i1   = idx_path1(t);          % current state index
    a1   = C1(i1);                 % optimal action index
    X_path1(t) = A(a1);            % current extraction
    % use Topt to find next state index
    [~, next_idx1, ~] = find(Topt1(i1,:), 1, 'first');
    if isempty(next_idx1), next_idx1 = 1; end  % for safety, though should not happen
    idx_path1(t+1) = next_idx1;
    S_path1(t+1)   = S(next_idx1);

    % if reached zero stock, set remaining extraction and stock to zero
    if idx_path1(t+1) == 1
        X_path1(t+1:end) = 0;
        S_path1(t+1:end) = 0;
    end

    % --- Utility function 2 ---
    i2   = idx_path2(t);
    a2   = C2(i2);
    X_path2(t) = A(a2);
    [~, next_idx2, ~] = find(Topt2(i2,:), 1, 'first');
    if isempty(next_idx2), next_idx2 = 1; end
    idx_path2(t+1) = next_idx2;
    S_path2(t+1)   = S(next_idx2);

    if idx_path2(t+1) == 1
        X_path2(t+1:end) = 0;
        S_path2(t+1:end) = 0;
    end
end

% final extraction at T_sim
X_path1(T_sim) = A(C1(idx_path1(T_sim)));
X_path2(T_sim) = A(C2(idx_path2(T_sim)));

% j) For both utility functions, plot out the optimal extraction path and the price path against time.
% Notice that they’re bumpy in spots rather than nice and smooth, even though we have 501 discrete states. 
% We’ll use interpolation to solve this problem in part 2.
% ---- Price paths from optimal extraction (inverse demand / marginal utility) ----
p1 = @(y) 1 ./ sqrt(y);     % u1'(y)
p2 = @(y) 5 - 0.1 .* y;     % u2'(y)

% To avoid division by zero or negative prices, set a small epsilon
epsy = 1e-12;
P_path1 = p1(max(X_path1, epsy));
P_path1(X_path1 == 0) = NaN;
P_path2 = p2(X_path2);

time = 1:T_sim;
figure;
subplot(2,2,1);
plot(time, X_path1, '-o');
title('Task 1 - Optimal Extraction Path - Utility Function 1');
xlabel('Time Period');
ylabel('Extraction Amount');
subplot(2,2,2);
plot(time, P_path1, '-o');
title('Task 1 - Price Path - Utility Function 1');
xlabel('Time Period'); 
ylabel('Price');
subplot(2,2,3);
plot(time, X_path2, '-o');
title('Task 1 - Optimal Extraction Path - Utility Function 2');
xlabel('Time Period');
ylabel('Extraction Amount');
subplot(2,2,4);
plot(time, P_path2, '-o');
title('Task 1 - Price Path - Utility Function 2');
xlabel('Time Period'); 
ylabel('Price');

saveas(gcf, '/users/tangheng/downloads/energy econ I material/Compute_task/task1_extraction_price_paths.png');


% -------------Task 2: Interpolating between the states.-------------
% a) Set up the state space as you did in part 1. We’ll define the action space differently here.
% Set up the action space so that it still runs from 0 to 1000 over 501 steps, 
% but now make the step size even over the square root of the action space.
clc;
clear;
S_tot = 1000;
N = 501;
S = linspace(0, S_tot, N)'; % State space vector
A_sqrt = linspace(0, sqrt(S_tot), N)'; % Action space over square root
A = A_sqrt .^ 2; % Action space vector
nA = N; % Number of possible actions
delta = 1/(1 + 0.05); % Discount factor
u1 = @(x) 2 * x .^ 0.5; % Utility function 1
u2 = @(x) 5 * x - 0.05 * x .^ 2; % Utility function 2
U1 = -inf(N, nA); % Flow utility matrix for utility function 1
U2 = -inf(N, nA); % Flow utility matrix for utility function 2
for i = 1:N
    for j = 1:nA
        if A(j) <= S(i)
            U1(i, j) = u1(A(j));
            U2(i, j) = u2(A(j));
        end
    end
end
next_state_index = zeros(N, nA); % Next state index matrix
for i = 1:N
    for j = 1:nA
        remaining_stock = S(i) - A(j);
        if remaining_stock < 0
            next_state_index(i, j) = 1; % Zero stock state
        else
            [~, idx] = min(abs(S - remaining_stock));
            next_state_index(i, j) = idx;
        end
    end
end

% b) Figuring out the state transition matrix T.
% What we want to do now is interpolate. Suppose next period’s state, for a given entering state and action, is 502.5.
% Your code needs to find the indices for the nearest two states (502 and 504), and then assign probability weights to each of those states.
% In this example the weights would be 0.75 on 502 and 0.25 on 504 (simple linear interpolation). 
T = sparse(N, N * nA); % State transition matrix
for j = 1:nA
    for i = 1:N
        remaining_stock = S(i) - A(j);
        if remaining_stock < 0
            T(i, (j - 1) * N + 1) = 1; % Zero stock state
        else
            lower_idx = find(S <= remaining_stock, 1, 'last');
            upper_idx = find(S >= remaining_stock, 1, 'first');
            if lower_idx == upper_idx
                T(i, (j - 1) * N + lower_idx) = 1;
            else
                weight_upper = (remaining_stock - S(lower_idx)) / (S(upper_idx) - S(lower_idx));
                weight_lower = 1 - weight_upper;
                T(i, (j - 1) * N + lower_idx) = weight_lower;
                T(i, (j - 1) * N + upper_idx) = weight_upper;
            end
        end
    end
end

% c) Solve the model using value function iteration as you did in part 1.
tolerance = 1e-8;
max_iterations = 10000;
V1 = zeros(N, 1); % Value function for utility function 1
V2 = zeros(N, 1); % Value function for utility function 2
C1 = zeros(N, 1); % Control function for utility function 1
C2 = zeros(N, 1); % Control function for utility function 2
for iter = 1:max_iterations
    Vnext1 = zeros(N, nA);
    Vnext2 = zeros(N, nA);
    for j = 1:nA
        Vnext1(:, j) = T(:, (j - 1) * N + 1:j * N) * V1;
        Vnext2(:, j) = T(:, (j - 1) * N + 1:j * N) * V2;
    end
    Vnew1 = U1 + delta * Vnext1;
    Vnew2 = U2 + delta * Vnext2;
    [V1_new, C1_new] = max(Vnew1, [], 2);
    [V2_new, C2_new] = max(Vnew2, [], 2);
    if max(abs(V1_new - V1)) < tolerance && max(abs(V2_new - V2)) < tolerance
        V1 = V1_new; C1 = C1_new;
        V2 = V2_new; C2 = C2_new;
        break;
    end
    V1 = V1_new; C1 = C1_new;
    V2 = V2_new; C2 = C2_new;
end
% finding the optimal actions.
X1_opt = A(C1);
X2_opt = A(C2);
% finding the optimal state transitions.
Topt1 = sparse(N, N);
Topt2 = sparse(N, N);
for i = 1:N
    j1 = C1(i);                          % selected optimal action index for state i
    j2 = C2(i);
    cols1 = (j1 - 1)*N + (1:N);          % corresponding columns in T for action j1 j2
    cols2 = (j2 - 1)*N + (1:N);
    % extract the row and assign to Topt
    Topt1(i, :) = T(i, cols1);
    Topt2(i, :) = T(i, cols2);
end

% d) Simulate the model for t = 80 periods, starting with the initial stock of 1000.
% with linear interpolation of the actions between grid points.
interp_policy = @(Sgrid, Xopt, s) ...
    min( interp1(Sgrid, Xopt, min(max(s, Sgrid(1)), Sgrid(end)), 'linear'), s );

T_sim  = 80;

S_path1 = zeros(T_sim,1);  X_path1 = zeros(T_sim,1);
S_path2 = zeros(T_sim,1);  X_path2 = zeros(T_sim,1);

S_path1(1) = S_tot;
S_path2(1) = S_tot;

for t = 1:T_sim-1
    % ===== Utility 1 =====
    s1 = S_path1(t);
    x1 = interp_policy(S, X1_opt, s1);
    X_path1(t)   = x1;
    S_path1(t+1) = max(0, s1 - x1);
    if S_path1(t+1) == 0
        % reached zero stock, set remaining extraction and stock to zero
        X_path1(t+1:end) = 0;  S_path1(t+1:end) = 0;
    end

    % ===== Utility 2 =====
    s2 = S_path2(t);
    x2 = interp_policy(S, X2_opt, s2);
    X_path2(t)   = x2;
    S_path2(t+1) = max(0, s2 - x2);
    if S_path2(t+1) == 0
        X_path2(t+1:end) = 0;  S_path2(t+1:end) = 0;
    end
end

% final extraction at T_sim
if S_path1(end) > 0
    X_path1(end) = interp_policy(S, X1_opt, S_path1(end));
end
if S_path2(end) > 0
    X_path2(end) = interp_policy(S, X2_opt, S_path2(end));
end

% e) For both utility functions, plot out the optimal extraction path and the price path against time.
p1 = @(y) 1 ./ sqrt(y);     % u1'(y)
p2 = @(y) 5 - 0.1 .* y;     % u2'(y)
epsy = 1e-12;
P_path1 = p1(max(X_path1, epsy));
P_path1(X_path1 == 0) = NaN;
P_path2 = p2(X_path2);
time = 1:T_sim;
figure;
subplot(2,2,1);
plot(time, X_path1, '-o');
title('Task 2 - Optimal Extraction Path - Utility Function 1');
xlabel('Time Period');
ylabel('Extraction Amount');
subplot(2,2,2);
plot(time, P_path1, '-o');
title('Task 2 - Price Path - Utility Function 1');
xlabel('Time Period'); 
ylabel('Price');
subplot(2,2,3);
plot(time, X_path2, '-o');
title('Task 2 - Optimal Extraction Path - Utility Function 2');
xlabel('Time Period');
ylabel('Extraction Amount');
subplot(2,2,4);
plot(time, P_path2, '-o');
title('Task 2 - Price Path - Utility Function 2');
xlabel('Time Period'); 
ylabel('Price');

saveas(gcf, '/users/tangheng/downloads/energy econ I material/Compute_task/task2_extraction_price_paths.png');


% ---------------- Task 3 (Use the FOC)  ----------------
clc; clear;

S_tot = 1000;
T_sim = 80;
r     = 0.05;
% (a) FOC
x_u1_from_mu0 = @(mu0,t,r) (1./(mu0.^2)) .* (1+r).^(-2.*t);
x_u2_from_mu0 = @(mu0,t,r) max(0, min(50, 10.*(5 - mu0.*(1+r).^t)));

% (b) cumulative production functions
cumprod_u1 = @(mu0,T,r) sum( x_u1_from_mu0(mu0,(0:T-1)',r) );
cumprod_u2 = @(mu0,T,r) sum( x_u2_from_mu0(mu0,(0:T-1)',r) );

% (c) solve for mu0 that exhausts the resource stock S_tot over T_sim periods
% u1 use fzero with an initial guess
a = (1+r)^(-2);
sum_geom = (1 - a^T_sim) / (1 - a);
mu0_u1_guess = sqrt( 1 / (S_tot * sum_geom) );
mu0_u1 = fzero(@(m) cumprod_u1(m,T_sim,r) - S_tot, mu0_u1_guess);

% u2 use fzero over an interval
g = @(m) cumprod_u2(m,T_sim,r) - S_tot;
mu0_u2 = fzero(g, [1e-6, 4.999]);

t = (0:T_sim-1)';
X_path1 = x_u1_from_mu0(mu0_u1, t, r);
X_path2 = x_u2_from_mu0(mu0_u2, t, r);

figure;
subplot(2,1,1); plot(t, X_path1, '-o');
title('Task 3 - Optimal Extraction Path - Utility Function 1');
xlabel('Time Period'); ylabel('Extraction Amount');

subplot(2,1,2); plot(t, X_path2, '-o');
title('Task 3 - Optimal Extraction Path - Utility Function 2');
xlabel('Time Period'); ylabel('Extraction Amount');
saveas(gcf, '/users/tangheng/downloads/energy econ I material/Compute_task/task3_extraction_paths.png');


% -------------Task 4: Drilling and real options.-------------

% a) Set up the state space as a vector of possible oil prices. 
% Let prices vary from $0 to $80, with a step size of $1 (so that the number of states N = 81). 
% Define a vector that returns the firm’s profits from drilling as a function of the price.
clc;
clear;

S_min = 0;  S_max = 80;  step_size = 1;
S = (S_min:step_size:S_max)';          % 81×1
N = numel(S);

D = 3e6;                                % drillling cost
X = 1e5;                                % barrels produced
profits = @(P) P .* X - D;              % instant profits from drilling

profits_vec = profits(S);               % 81×1 vector of profits
P_break = D / X;                        % break-even price, 30 dollars

% b) Define a state transition matrix T.
% For cutoffs, use the price halfway between states (so, for example, 
% a next period’s price of $49.51 will be assigned to the $50 price state, 
% whereas a price of $49.49 would be assigned to the $49 price state).
% For the lowest price, the bottom cutoff is -∞, and for the highest price use a top cutoff of +∞.
sigma = 4;                             % standard deviation of price shock
T = sparse(N, N);                      % state transition matrix
for i = 1:N
    P_current = S(i);
    for j = 1:N
        if j == 1
            lower_bound = -inf;
        else
            lower_bound = (S(j-1) + S(j)) / 2;
        end
        if j == N
            upper_bound = inf;
        else
            upper_bound = (S(j) + S(j+1)) / 2;
        end
        mu = P_current; % expected next period price
        prob = normcdf(upper_bound, mu, sigma) - normcdf(lower_bound, mu, sigma);
        T(i, j) = prob;
    end
end
% verify each row sums to 1
row_sums = sum(T, 2);
disp('Max row sum deviation from 1:');
disp(max(abs(row_sums - 1)));

% c) Use value function iteration to compute the value function and the control function.
delta = 1/1.05;

V = zeros(N,1);
tolerance = 1e-8;
maxit = 10000;

for it = 1:maxit
    V_wait  = delta * (T * V);     % wait：δ·E[V_{t+1}]
    V_drill = profits_vec;         % drill：get PX-D

    V_new = max(V_drill, V_wait);  % Bellman
    if max(abs(V_new - V)) < tolerance
        V = V_new; break;
    end
    V = V_new;
end

drill_now = (V_drill >= V_wait);   % N×1 logical vector
% find the trigger price
if any(drill_now)
    trigger_price = S(find(drill_now,1,'first'));
else
    trigger_price = NaN;
end

disp('Optimal Trigger Price:');
disp(trigger_price);

% d) Plot the value function as a function of the oil price.
figure;
plot(S, V, '-o');
title('Task 4 - Value Function of Drilling Option');
xlabel('Oil Price ($/barrel)');
ylabel('Value of Drilling Option ($)');
saveas(gcf, '/users/tangheng/downloads/energy econ I material/Compute_task/task4_value_function.png');