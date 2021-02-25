clear 
close all
clc

%% Generate NN parameters
rng('default');

% Fix issue with sedumi being removed from path
addpath('cvx\sedumi')  

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
W = net.weights;
b = net.biases;



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

% Activation functions bounds
alpha = 0;
beta = 0.5;

v1 = W{1}*u + b{1};
con_hid11a = x11 - alpha;
con_hid11b = x11 - beta*v1(1);
con_hid11c = x11*(x11 - v1(1));

con_hid12a = x12 - alpha;
con_hid12b = x12 - beta*v1(2);
con_hid12c = x12*(x12 - v1(2));

v2 = W{2}*[x11;x12] + b{2};
con_hid21a = x21 - alpha;
con_hid21b = x21 - beta*v2(1);
con_hid21c = x21*(x21 - v2(1));

con_hid22a = x22 - alpha;
con_hid22b = x22 - beta*v2(2);
con_hid22c = x22*(x22 - v2(2));

% Output layer constraints
v3 = W{3}*[x21;x22] + b{3};
c = -1; % c = 1 is max and c = -1 is minimum
f = -c*(y_lim - v3); % Minus sign can be used to check maximium

% Polynomials for cone
for j = 1:13
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0:2));
    [prog] = sosineq(prog,s{j});
end

% Polynomials for ideal
for j = 1:4
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0:2));
end

% Create statement 1 + cone + ideal
expr = - f;

% expr = expr - s{1}*con_in1*con_in2 - s{2}*con_hid11a*con_hid11b - s{3}*con_hid12a*con_hid12b ...
%                                  - s{4}*con_hid21a*con_hid21b - s{5}*con_hid22a*con_hid22b;

%expr = expr - s{1}*con_in1*con_in2 - s{2}*con_hid11a*con_hid12a - s{3}*con_hid11b*con_hid12b ...
%                                 - s{4}*con_hid21a*con_hid22a - s{5}*con_hid21b*con_hid22b;

% expr = expr - s{1}*con_in1 - s{2}*con_in2 - s{3}*con_hid11a - s{4}*con_hid11b - s{5}*con_hid12a ...
%       - s{6}*con_hid12b - s{7}*con_hid21a - s{8}*con_hid21b - s{9}*con_hid22a - s{10}*con_hid22b;
% 
% expr = expr - s{11}*con_in1*con_in2 ...
% - s{12}*con_hid11a*con_hid11b - s{13}*con_hid12a*con_hid12b - s{14}*con_hid21a*con_hid21b - s{15}*con_hid22a*con_hid22b;

expr = expr - s{1}*con_in1 - s{2}*con_in2 - s{3}*con_hid11b - s{4}*con_hid12b - s{5}*con_hid21b - s{6}*con_hid22b;

expr = expr - s{7}*con_in1*con_in2;

expr = expr - s{8}*con_hid11b*con_hid12b - s{9}*con_hid11b*con_hid21b - s{10}*con_hid11b*con_hid22b ...
    - s{11}*con_hid12b*con_hid21b - s{12}*con_hid12b*con_hid22b - s{13}*con_hid21b*con_hid22b;
%expr = expr - s{14}*con_in1*con_hid11b - s{15}*con_in1*con_hid12b - s{16}*con_in1*con_hid21b - s{17}*con_in1*con_hid22b;
%expr = expr - s{18}*con_in2*con_hid11b - s{19}*con_in2*con_hid12b - s{20}*con_in2*con_hid21b - s{21}*con_in2*con_hid22b;

%expr = expr - s{18}*con_in1*con_in2*con_hid11b - s{19}*con_in2*con_hid12b - s{20}*con_in2*con_hid21b - s{21}*con_in2*con_hid22b;
                                
expr = expr - t{1}*con_hid11c - t{2}*con_hid12c - t{3}*con_hid21c - t{4}*con_hid22c;

% P-satz refutation condition
prog = sosineq(prog,expr);
prog = sossetobj(prog,c*y_lim);
solver_opt.solver = 'sedumi';
prog = sossolve(prog,solver_opt);
sosgetsol(prog,y_lim)

%% Faz method

% Solver options
options.language = 'cvx';
options.solver = 'sedumi';
options.verbose = false;
method = 'deepsdp';

% Run DeepSDP
repeated = 0;
%[X_SDP,Y_SDP,chordal_test(i,:)] = output_polytope(net,x_min,x_max,method,repeated,options,m);
c = 1;
[bound, ~,~] = deep_sdp(net,u_min,u_max,c,repeated,options);
DSDP_max = bound

c = -1;
[bound, ~,~] = deep_sdp(net,u_min,u_max,c,repeated,options);
DSDP_min = -bound

%% Evaluate NN through computation
Xin = linspace(u_min,u_max,500000);
Y = Xin;
num_layers = length(dim_hidden);
for l = 1:num_layers
     Y = max(0,W{l}*Y + repmat(b{l}(:),1,size(Y,2)));
end
Y = W{end}*Y + repmat(b{end}(:),1,size(Y,2));
Xout = Y;

true_min = min(Y)
true_max = max(Y)

%% Ignore
%data = scatter(Xin, Xout);

%data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');

% 
% Xin = rect2d(u_min,u_max);
% Xout = net.eval(Xin);
% data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');hold on;


% 
% % Inequaity constraints
% constraints{1} = con_in1; constraints{2} = con_in2;
% constraints{3} = con_hid11a; constraints{4} = con_hid11b;
% constraints{5} = con_hid12a; constraints{6} = con_hid12b;
% constraints{7} = con_hid21a; constraints{8} = con_hid21b;
% constraints{9} = con_hid22a; constraints{10} = con_hid22b;
% 
% % Equality constraints
% constraints2{1} = con_hid11c; constraints2{2} = con_hid12c;
% constraints2{3} = con_hid21c; constraints2{4} = con_hid22c;


%expr = expr - s{1}*con_in1*con_in2;
% expr = expr - s{1}*con_in1 - s{2}*con_in2;
% for j = 3:10
%     expr = expr - s{j}*(constraints{j});
% end
% 
% for j = 1:4
%     expr = expr - t{j}*constraints2{j};
% end

