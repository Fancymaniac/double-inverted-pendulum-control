%% Digital Control of a Double Inverted Pendulum System

clear; close all; clc;

%% 1. System Parameters and Continuous-Time Model
M = 1.0;        % Cart mass (kg)
m1 = 0.5;       % First pendulum mass (kg)
m2 = 0.2;       % Second pendulum mass (kg)
L1 = 0.5;       % First pendulum length (m)
L2 = 0.3;       % Second pendulum length (m)
g = 9.81;       % Gravity (m/s^2)
b = 0.1;        % Damping coefficient for the cart

% Derived parameters
l1 = L1/2;      % Center of mass for pendulum 1
l2 = L2/2;      % Center of mass for pendulum 2
J1 = (1/12)*m1*L1^2;  % Moment of inertia for pendulum 1
J2 = (1/12)*m2*L2^2;  % Moment of inertia for pendulum 2

% Continuous-time state-space model
% State vector: x = [cart; cart_dot; theta1; theta1_dot; theta2; theta2_dot]
A = [0,      1,           0,         0,           0,         0;
     0, -b/M, (m1*g*l1)/M,         0,  (m2*g*l2)/M,         0;
     0,      0,           0,         1,           0,         0;
     0, -b/(M*L1), -(M+m1)*g/(M*L1),   0, -m2*g*l2/(M*L1),   0;
     0,      0,           0,         0,           0,         1;
     0, -b/(M*L2), -m1*g*l1/(M*L2),   0, -(M+m2)*g/(M*L2),   0];
B = [0; 1/M; 0; 1/(M*L1); 0; 1/(M*L2)];
C = [1, 0, 0, 0, 0, 0;    % Measure cart position
     0, 0, 1, 0, 0, 0;    % Measure first pendulum angle
     0, 0, 0, 0, 1, 0];   % Measure second pendulum angle
D = zeros(3,1);

% Display continuous-time eigenvalues
disp('Continuous-time system eigenvalues:');
disp(eig(A));

%% 2. Discretization using Zero-Order Hold
T = 0.01;  % Sampling time (s)
sys_c = ss(A, B, C, D);
sys_d = c2d(sys_c, T, 'zoh');
G = sys_d.A;  % Discrete state matrix
H = sys_d.B;  % Discrete input matrix
Cd = sys_d.C; % Discrete output matrix
Dd = sys_d.D; % Discrete direct transmission

disp('Discrete-time state matrix (G):');
disp(G);
disp('Discrete-time input matrix (H):');
disp(H);
disp('Discrete-time system eigenvalues:');
disp(eig(G));

%% 3. Controller Design: LQR with Integral Action
% Define weighting matrices for original system and R for control
Q = diag([100, 10, 500, 50, 500, 50]);
R = 0.1;

% Augment the system with an integrator for cart position tracking.
% Let x1 (cart position) be measured by C1.
C1 = [1, 0, 0, 0, 0, 0];
Ga = [G, zeros(6,1); -C1*T, 1];
Ha = [H; 0];

% Augmented weighting matrices: high weight on integrator state (200)
Qa = blkdiag(Q, 200);
Ra = R;

% Compute LQR gains for the augmented system
[Ka, S, e_lqr] = dlqr(Ga, Ha, Qa, Ra);
K_sf = Ka(1:end-1);   % State feedback gains for 6 states
Ki   = Ka(end);       % Integrator gain

disp('LQR with Integral Action Gains:');
disp('State-feedback gains (K_sf):');
disp(K_sf);
disp('Integral gain (Ki):');
disp(Ki);

%% 4. Observer Design: Luenberger Observer
% For observer design, we use the same discrete-time system.
% Define desired continuous-time observer poles.
p_cont = [-2, -2.1, -2.2, -2.3, -2.4, -2.5];
% Map to discrete time and make them 2x faster:
p_obs = exp(2 * p_cont * T);
% Compute observer gain:
K_e = place(G', Cd', p_obs)';
disp('Luenberger Observer Gain (K_e):');
disp(K_e);

%% 4.4 Kalman Filter Implementation
% Define noise covariance matrices:
Q_kf = diag([0.01, 0.1, 0.01, 0.1, 0.01, 0.1]);  % Process noise covariance
R_kf = diag([0.01, 0.01, 0.01]);                  % Measurement noise covariance

% Initialize Kalman filter variables:
x_kf = zeros(6, 1);      % Initial state estimate
P_kf = eye(6);           % Initial error covariance

%% 5. Simulation Setup
t_final = 5;                   % Total simulation time (s)
t = 0:T:t_final;               % Time vector
N = length(t);

% Initial state: small initial angles and zero velocities for cart
x0 = [0; 0; 0.1; 0; 0.05; 0];
x = zeros(6, N);   x(:,1) = x0;        % True state
x_hat = zeros(6, N); x_hat(:,1) = zeros(6,1);  % Observer state estimate
x_kf_store = zeros(6, N);              % Kalman filter state estimate
u = zeros(1, N);                     % Control input
y_meas = zeros(3, N);                % Measured output
e_int = 0;                           % Integral error (for LQR with integrator)

% Reference trajectory: 1-second ramp to 0.5m, then constant
idx_ramp = round(1/T);
r_ref = zeros(N, 1);
r_ref(1:idx_ramp) = linspace(0, 0.5, idx_ramp);
r_ref(idx_ramp+1:end) = 0.5;

%% 6. Simulation Loop (Nominal System)
for k = 1:N-1
    % True plant output:
    y_meas(:,k) = Cd * x(:,k);
    
    % (Optional) Add measurement noise here if desired:
    % y_noisy = y_meas(:,k) + 0.01*randn(3,1);
    % For now, use y_meas directly:
    
    % Update integral error for LQR:
    e_int = e_int + (r_ref(k) - y_meas(1,k)) * T;
    
    % Observer update (Luenberger):
    x_hat(:,k+1) = G*x_hat(:,k) + H*u(k) + K_e*(y_meas(:,k) - Cd*x_hat(:,k));
    
    % Kalman filter prediction step:
    x_pred = G * x_kf + H * u(k);
    P_pred = G * P_kf * G' + Q_kf;
    % Kalman gain:
    K_kf = P_pred * Cd' / (Cd * P_pred * Cd' + R_kf);
    % Kalman filter update:
    x_kf = x_pred + K_kf * (y_meas(:,k) - Cd * x_pred);
    P_kf = (eye(6) - K_kf * Cd) * P_pred;
    x_kf_store(:, k+1) = x_kf;
    
    % Control law (LQR with integral action):
    u(k) = -K_sf * x_hat(:,k) + Ki * e_int;
    
    % Apply saturation to control input (±50 N):
    u(k) = max(min(u(k), 50), -50);
    
    % Plant update:
    x(:,k+1) = G * x(:,k) + H * u(k);
end
% Final measurement update:
y_meas(:,N) = Cd * x(:,N);

%% 7. Robustness Test: Parameter Variation (Increase m2 by 50%)
% Change m2 to 0.3 kg
m2_var = 0.3;
A_var = [0, 1, 0, 0, 0, 0;
         0, -b/M, (m1*g*l1)/M, 0, (m2_var*g*l2)/M, 0;
         0, 0, 0, 1, 0, 0;
         0, -b/(M*L1), -(M+m1)*g/(M*L1), 0, -m2_var*g*l2/(M*L1), 0;
         0, 0, 0, 0, 0, 1;
         0, -b/(M*L2), -m1*g*l1/(M*L2), 0, -(M+m2_var)*g/(M*L2), 0];
sys_c_var = ss(A_var, B, C, D);
sys_d_var = c2d(sys_c_var, T, 'zoh');
G_var = sys_d_var.A; H_var = sys_d_var.B;

x_var = zeros(6, N);   x_var(:,1) = x0;
y_var = zeros(3, N);
u_var = zeros(1, N);
e_int_var = 0;
for k = 1:N-1
    y_var(:,k) = Cd * x_var(:,k);
    e_int_var = e_int_var + (r_ref(k) - y_var(1,k)) * T;
    u_var(k) = -K_sf * x_var(:,k) + Ki * e_int_var;
    u_var(k) = max(min(u_var(k), 50), -50);
    x_var(:,k+1) = G_var * x_var(:,k) + H_var * u_var(k);
end
y_var(:,N) = Cd * x_var(:,N);

%% 8. Plotting Results

% Figure 1: Cart Position and Pendulum Angles Tracking
figure(1);
subplot(3,1,1);
plot(t, y_meas(1,:)', 'LineWidth', 1.5); hold on;
plot(t, r_ref, '--', 'LineWidth', 1.5);
title('Cart Position Tracking');
xlabel('Time (s)'); ylabel('Position (m)');
legend('Cart Position','Reference'); grid on;

subplot(3,1,2);
plot(t, y_meas(2,:)', 'LineWidth', 1.5);
title('First Pendulum Angle');
xlabel('Time (s)'); ylabel('Angle (rad)'); grid on;

subplot(3,1,3);
plot(t, y_meas(3,:)', 'LineWidth', 1.5);
title('Second Pendulum Angle');
xlabel('Time (s)'); ylabel('Angle (rad)'); grid on;

% Figure 2: Control Input
figure(2);
plot(t, u, 'LineWidth', 1.5);
title('Control Input');
xlabel('Time (s)'); ylabel('Force (N)');
grid on;

% Figure 3: Observer Performance (Estimation Error)
figure(3);
subplot(3,1,1);
plot(t, x(1,:) - x_hat(1,:), 'LineWidth', 1.5);
title('Cart Position Estimation Error');
xlabel('Time (s)'); ylabel('Error (m)'); grid on;
subplot(3,1,2);
plot(t, x(3,:) - x_hat(3,:), 'LineWidth', 1.5);
title('First Pendulum Angle Estimation Error');
xlabel('Time (s)'); ylabel('Error (rad)'); grid on;
subplot(3,1,3);
plot(t, x(5,:) - x_hat(5,:), 'LineWidth', 1.5);
title('Second Pendulum Angle Estimation Error');
xlabel('Time (s)'); ylabel('Error (rad)'); grid on;

% Figure 4: Nominal vs. Varied Parameter (m2)
figure(4);
subplot(3,1,1);
plot(t, y_meas(1,:)', 'LineWidth', 1.5); hold on;
plot(t, y_var(1,:)', '--', 'LineWidth', 1.5);
title('Cart Position: Nominal vs. Varied m2');
xlabel('Time (s)'); ylabel('Position (m)');
legend('Nominal','m2 = 0.3 kg'); grid on;
subplot(3,1,2);
plot(t, y_meas(2,:)', 'LineWidth', 1.5); hold on;
plot(t, y_var(2,:)', '--', 'LineWidth', 1.5);
title('First Pendulum Angle: Nominal vs. Varied m2');
xlabel('Time (s)'); ylabel('Angle (rad)');
legend('Nominal','m2 = 0.3 kg'); grid on;
subplot(3,1,3);
plot(t, y_meas(3,:)', 'LineWidth', 1.5); hold on;
plot(t, y_var(3,:)', '--', 'LineWidth', 1.5);
title('Second Pendulum Angle: Nominal vs. Varied m2');
xlabel('Time (s)'); ylabel('Angle (rad)');
legend('Nominal','m2 = 0.3 kg'); grid on;

% Figure 5: State Estimation Comparison (Cart Position)
figure(5);
plot(t, x(1,:)', 'LineWidth', 1.5); hold on;
plot(t, x_hat(1,:)', '--', 'LineWidth', 1.5);
plot(t, x_kf_store(1,:)', '-.', 'LineWidth', 1.5);
title('State Estimation Comparison: Cart Position');
xlabel('Time (s)'); ylabel('Position (m)');
legend('True State','Luenberger Observer','Kalman Filter');
grid on;

%% End of Script