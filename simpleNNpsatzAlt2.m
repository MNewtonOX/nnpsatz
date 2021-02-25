clear 
close all
clc

%% Generate NN parameters
rng('default');

% Number of edges of safety polytope
m = 6;

% Input dimension
dim_in = 1; %u

% Input bounds
u_min = -5;
u_max = 5;

% Hidden layer dimensions
dim_hidden = [2,2];

% Ouput dimension
dim_out = 1;

% Create NN parameters
dims = [dim_in, dim_hidden, dim_out];
net = nnsequential(dims,'relu');
%save(['net-' num2str(num_hidden_units_per_layer) 'n.mat'],'net');
W = net.weights;
b = net.biases;

% % Set parameters to simple values for easier analyisis 
% W{1} = [1;1]; b{1} = [0;0];
% W{2} = ones(2,2); b{2} = [0;0];
% W{3} = [1,1]; b{3} = 0;

% Activation functions bounds
alpha = 0;
beta = 1;

%% SOS 
% Start off hand crafted for small example, can make general later
syms u x11 x12 x21 x22 y_lim
vars = [u x11 x12 x21 x22]; % y_lim]; 
prog = sosprogram(vars);
prog = sosdecvar(prog,y_lim);

% Input constraints
con_in1 = u - u_min;
con_in2 = u_max - u;

% Hidden layer constraints
v1 = W{1}*u + b{1};
con_hid11a = x11;
con_hid11b = x11 - v1(1);
con_hid11c = x11*(x11 - v1(1));

con_hid12a = x12;
con_hid12b = x12 - v1(2);
con_hid12c = x12*(x12 - v1(2));

v2 = W{2}*[x11;x12] + b{2};
con_hid21a = x21;
con_hid21b = x21 - v2(1);
con_hid21c = x21*(x21 - v2(1));

con_hid22a = x22;
con_hid22b = x22 - v2(2);
con_hid22c = x22*(x22 - v2(2));

% Output layer constraints
%y_max = -10; % How big the bound is
%c = [cos(0);sin(0)];
%y_lim = 10;
v3 = W{3}*[x21;x22] + b{3};
con_out = (y_lim - v3); % Minus sign can be used to check maximium
%con_out = (4.3 - v3)*(v3 - 3.7) % quadratic constraint

% Inequaity constraints
constraints{1} = con_in1; constraints{2} = con_in2;
constraints{3} = con_hid11a; constraints{4} = con_hid11b;
constraints{5} = con_hid12a; constraints{6} = con_hid12b;
constraints{7} = con_hid21a; constraints{8} = con_hid21b;
constraints{9} = con_hid22a; constraints{10} = con_hid22b;

% Equality constraints
constraints2{1} = con_hid11c; constraints2{2} = con_hid12c;
constraints2{3} = con_hid21c; constraints2{4} = con_hid22c;

% P-sat refutation
f = -con_out; 

% Polynomials for cone
for j = 1:10
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0:2));
    [prog] = sosineq(prog,s{j});
end

% Polynomials for ideal
for j = 1:4
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0:2));
end

% Create statement 1 + cone + ideal
%expr = 1 + s{1} + s{2}*f;
% Rearrange to be in terms of sos, s{1} is sos, s{2} = 1
expr = - 1 - f;
%expr = - f; % When I drop the 1 here the bounds become tigher, not sure if this is because of chance or that we don't need the one

% expr = expr - s{1}*con_in1*con_in2 - s{2}*con_hid11a*con_hid11b - s{3}*con_hid12a*con_hid12b ...
%                                  - s{4}*con_hid21a*con_hid21b - s{5}*con_hid22a*con_hid22b;

expr = expr - s{1}*con_in1*con_in2 - s{2}*con_hid11a*con_hid12a - s{3}*con_hid11b*con_hid12b ...
                                 - s{4}*con_hid21a*con_hid22a - s{5}*con_hid21b*con_hid22b;

% expr = expr - s{1}*con_in1 - s{2}*con_in2 - s{3}*con_hid11a - s{4}*con_hid11b - s{5}*con_hid12a ...
%     - s{6}*con_hid12b - s{7}*con_hid21a - s{8}*con_hid21b - s{9}*con_hid22a - s{10}*con_hid22b;
                               
expr = expr - t{1}*con_hid11c - t{2}*con_hid12c - t{3}*con_hid21c - t{4}*con_hid22c;

%expr = expr - s{1}*con_in1*con_in2;
% expr = expr - s{1}*con_in1 - s{2}*con_in2;
% for j = 3:10
%     expr = expr - s{j}*(constraints{j});
% end
% 
% for j = 1:4
%     expr = expr - t{j}*constraints2{j};
% end

% P-satz refutation condition
prog = sosineq(prog,expr);
prog = sossetobj(prog,y_lim);
solver_opt.solver = 'sedumi';
prog = sossolve(prog,solver_opt);
sosgetsol(prog,y_lim)

% Evaluate NN through computation
Xin = linspace(u_min,u_max,50);
Y = Xin;
num_layers = length(dim_hidden);
for l = 1:num_layers
     Y = max(0,W{l}*Y + repmat(b{l}(:),1,size(Y,2)));
end
Y = W{end}*Y + repmat(b{end}(:),1,size(Y,2));
Xout = Y;

true_min = min(Y)
true_max = max(Y)
%data = scatter(Xin, Xout);

%data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');

% 
% Xin = rect2d(u_min,u_max);
% Xout = net.eval(Xin);
% data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');hold on;




