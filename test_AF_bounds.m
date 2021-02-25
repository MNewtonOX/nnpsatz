% Test bounds on activation functions


%% tanh
syms x
fplot(tanh(x));
hold on
%fplot(diff(tanh(x)));

%fplot(1);
%fplot(-1);

fplot(0);
fplot(x);

y_min = -0.5;
y_max = 0.9;
x_min = tanh(y_min);
x_max = tanh(y_max);

m = (1 - tanh(y_max)^2 - (1 - tanh(y_min)^2))/(y_max - y_min);
c = (1 - tanh(y_min)^2) - m*y_min;

fplot((1 - m*x - c)^0.5);
fplot(-(1 - m*x - c)^0.5);

fplot(x_min);
fplot(x_max);

%% sigmoid
syms x
fplot(sig(x));
hold on
%fplot(difsig(x))

fplot(0)
fplot(1)

fplot(0.5)
fplot(0.25*x + 0.5)

y_min = -2;
y_max = 3;
x_min = sig(y_min);
x_max = sig(y_max);

fplot(x_min)
fplot(x_max)

m = (difsig(y_max) - difsig(y_min))/(y_max - y_min);
c = difsig(y_min) - m*y_min;
%fplot(m*x+c)
fplot(0.5 + (0.25 - m*x - c)^0.5)
fplot(0.5 - (0.25 - m*x - c)^0.5)

% % Function limits
% ineq_constraints{icount} = x_curr_layer(k) - alpha; icount = icount + 1;
% ineq_constraints{icount} = -x_curr_layer(k) + beta; icount = icount + 1;
% ineq_constraints{icount} = (x_curr_layer(k) - alpha)*(-x_curr_layer(k) + beta); icount = icount + 1;
% 
% % Sector constraints
% ineq_constraints{icount} = (x_curr_layer(k) - 0.5)*(v{j}(k)/4 + 0.5 - x_curr_layer(k)); icount = icount + 1;
% 
% % Derivative constraints
% ineq_constraints{icount} = x_curr_layer(k)*(1 - x_curr_layer(k)); icount = icount + 1;
% ineq_constraints{icount} = 0.25 - x_curr_layer(k)*(1 - x_curr_layer(k)); icount = icount + 1;
% 
% % Find lower bound line
% gradl = (difsig(Y_max_curr_layer(k)) - difsig(Y_min_curr_layer(k)))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
% cl = difsig(Y_min_curr_layer(k)) - gradl*Y_min_curr_layer(k);
% 
% ineq_constraints{icount} = x_curr_layer(k)*(1 - x_curr_layer(k)) - (gradl*v{j}(k) + cl) ; icount = icount + 1;
% 
% % Pre-processing bounds
% ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;

function [y] = difsig(x)
    y = sig(x)*(1 - sig(x));
end

function [y] = sig(x)
    y = 1./(1+exp(-x));
end

% ineq_constraints{icount} = x_curr_layer(k) + 1; icount = icount + 1;
%         ineq_constraints{icount} = -x_curr_layer(k) + 1; icount = icount + 1;
%         ineq_constraints{icount} = (x_curr_layer(k) + 1)*(-x_curr_layer(k) + 1); icount = icount + 1;
%         
%         % Sector constraints
%         ineq_constraints{icount} = (x_curr_layer(k) - 0)*(v{j}(k) - x_curr_layer(k)); icount = icount + 1;
%         
%         % Derivative constraints
%         ineq_constraints{icount} = x_curr_layer(k)^2; icount = icount + 1;
%         ineq_constraints{icount} = 1 - x_curr_layer(k)^2; icount = icount + 1;
%         
%         % Find lower bound line
%         gradl = (diftanh(Y_max_curr_layer(k)) - diftanh(Y_min_curr_layer(k)))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
%         cl = diftanh(Y_min_curr_layer(k)) - gradl*Y_min_curr_layer(k);
%         
%         ineq_constraints{icount} = 1 - x_curr_layer(k)^2 - (gradl*v{j}(k) + cl) ; icount = icount + 1;
%         
%         % Second derivative constraints
%         second_diff = 2*x_curr_layer(k)*(x_curr_layer(k)^2 - 1);
%         ineq_constraints{icount} = second_diff + (4*3^(1/2))/9; icount = icount + 1;
%         ineq_constraints{icount} = -second_diff + (4*3^(1/2))/9; icount = icount + 1;
%         
%         % Pre-processing bounds
%         ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;
