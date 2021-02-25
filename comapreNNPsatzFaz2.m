clear 
close all
clc

addpath('cvx\sedumi')  

%% Generate NN parameters
rng('default');

% Number of edges of safety polytope
m = 6;

% Input dimension
dim_in = 2; %u

% Input bounds
u_min = -5*ones(dim_in,1);
u_max = 5*ones(dim_in,1);

% Hidden layer dimensions
dim_hidden = [2,3,3,2];
%dim_hidden = [20];

% Ouput dimension
dim_out = 1;

% Create NN parameters
dims = [dim_in, dim_hidden, dim_out];
net = nnsequential(dims,'relu');
%save(['net-' num2str(num_hidden_units_per_layer) 'n.mat'],'net');
W = net.weights;
b = net.biases;

% Activation functions bounds
alpha = 0;
beta = 1;

%% SOS 
% Create symbolic variables
syms u [dim_in,1]
syms x [sum(dim_hidden),1]
syms y [dim_out,1]

vars = [u; x]; %[u x11 x12 x21 x22 y];
prog = sosprogram(vars);
prog = sosdecvar(prog,y);

% Input constraints
con_in1 = u - u_min;
con_in2 = u_max - u;

% Hidden layer constraints

% Pre-processing step to find active and inactive neurons
[Y_min,Y_max,X_min,X_max,~,~] = net.interval_arithmetic(u_min,u_max);
Ip = find(Y_min>1e-3);
In = find(Y_max<-1e-3);
Ipn = setdiff(1:sum(dim_hidden),union(Ip,In));

% Create cell for constraints
ineq_constraints = {};
eq_constraints = {};
icount = 1; %inequality
ecount = 1; %equality
for j = 1:length(dim_hidden)
    if j == 1
        x_prev_layer = u;
        x_curr_layer = x(1:dim_hidden(j));
        v{j} = W{j}*u + b{j};
        X_min_curr_layer = X_min(1:dim_hidden(j));
        X_max_curr_layer = X_max(1:dim_hidden(j));
    else
        x_prev_layer = x(sum(dim_hidden(1:j-2)) + 1 : sum(dim_hidden(1:j-1)));
        x_curr_layer = x(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        v{j} = W{j}*x_prev_layer + b{j};
        X_min_curr_layer = X_min(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        X_max_curr_layer = X_max(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
    end
    for k = 1:dim_hidden(j)
        node_num = sum(dim_hidden(1:j-1)) + k;
%         if X_min_curr_layer(k) > alpha
%             low_bound = X_min_curr_layer(k);
%         else 
%             low_bound = alpha;
%         end
        %if X_max_curr_layer(k) < beta
        if any(node_num == Ip) 
            ineq_constraints{icount} = x_curr_layer(k) - alpha*v{j}(k); icount = icount + 1;
            %ineq_constraints{icount} = (x_curr_layer(k) - alpha*v{j}(k))^2; icount = icount + 1;
            eq_constraints{ecount} = x_curr_layer(k) - beta*v{j}(k); ecount = ecount + 1;
            eq_constraints{ecount} = (x_curr_layer(k) - beta*v{j}(k))*(x_curr_layer(k) - beta*v{j}(k)); ecount = ecount + 1;
            %eq_constraints{ecount} = (x_curr_layer(k) - beta*v{j}(k))^4; ecount = ecount + 1;
        elseif any(node_num == In) 
            eq_constraints{ecount} = x_curr_layer(k) - alpha*v{j}(k); ecount = ecount + 1;
            ineq_constraints{icount} = x_curr_layer(k) - beta*v{j}(k); icount = icount + 1;
            %ineq_constraints{icount} = (x_curr_layer(k) - beta*v{j}(k))^2; icount = icount + 1;
            eq_constraints{ecount} = (x_curr_layer(k) - alpha*v{j}(k))*(x_curr_layer(k) - alpha*v{j}(k)); ecount = ecount + 1;
            %eq_constraints{ecount} = (x_curr_layer(k) - alpha*v{j}(k))^4; ecount = ecount + 1;
        elseif any(node_num == Ipn) 
            ineq_constraints{icount} = x_curr_layer(k) - alpha*v{j}(k); icount = icount + 1;
            ineq_constraints{icount} = x_curr_layer(k) - beta*v{j}(k); icount = icount + 1;
            %ineq_constraints{icount} = (x_curr_layer(k) - alpha*v{j}(k))^2; icount = icount + 1;
            %ineq_constraints{icount} = (x_curr_layer(k) - beta*v{j}(k))^2; icount = icount + 1;
            eq_constraints{ecount} = (x_curr_layer(k) - alpha*v{j}(k))*(x_curr_layer(k) - beta*v{j}(k)); ecount = ecount + 1;
            %eq_constraints{ecount} = ((x_curr_layer(k) - alpha*v{j}(k))*(x_curr_layer(k) - beta*v{j}(k)))^2; ecount = ecount + 1;
        end
        %ineq_constraints{icount} = x_curr_layer(k) - X_min_curr_layer(k); icount = icount + 1;
        %ineq_constraints{icount} = -x_curr_layer(k) + X_max_curr_layer(k); icount = icount + 1;
        ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;
    end
end

% Output layer constraints
%c = [cos(0);sin(0)];
v_out = W{end}*x(end - dim_hidden(end) + 1 : end) + b{end};
con_out = y - v_out;

% P-sat refutation
c = 1; % c = 1 is max, c = -1 is min
f = -c*con_out; 

% Polynomials for cone
for j = 1:length(ineq_constraints)
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0));
    prog = sosineq(prog,s{j});
end

% Polynomials for ideal
for j = 1:length(eq_constraints)
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0));
end

% Create statement 1 + cone + ideal
expr = - f;

% Input layer constraints
for j = 1:dim_in
%     [prog,s_in1{j}] = sospolyvar(prog,monomials(vars,0));
%     prog = sosineq(prog,s_in1{j});
%     [prog,s_in2{j}] = sospolyvar(prog,monomials(vars,0));
%     prog = sosineq(prog,s_in2{j});
    [prog,s_in3{j}] = sospolyvar(prog,monomials(vars,0));
    prog = sosineq(prog,s_in3{j});
    %expr = expr - s_in1{j}*con_in1(j) - s_in2{j}*con_in2(j);
    expr = expr - s_in3{j}*con_in1(j)*con_in2(j);
end

% Hidden layer constraints
for j = 1:length(ineq_constraints)
    expr = expr - s{j}*ineq_constraints{j};
end

for j = 1:length(eq_constraints)
    expr = expr - t{j}*eq_constraints{j};
end

% P-satz refutation
prog = sosineq(prog,expr);
prog = sossetobj(prog,c*y);
solver_opt.solver = 'sedumi';
prog = sossolve(prog,solver_opt);
sosgetsol(prog,y)

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

%% Evaluate through computation
if dim_in == 1
    Xin  = linspace(u_min,u_max,50);
elseif dim_in == 2
    Xin = rect2d(u_min,u_max);
elseif dim_in == 3
    [X1,X2,X3] = ndgrid(linspace(u_min(1),u_max(1),50),linspace(u_min(2),u_max(2),50),linspace(u_min(3),u_max(3),50));
    Xin(1,:) = X1(:);
    Xin(2,:) = X2(:);
    Xin(3,:) = X3(:);
elseif dim_in >= 4
    error('input size too large for now')
end

Xout = net.eval(Xin);
true_max = max(Xout)
true_min = min(Xout)

%% Junk
% Y = Xin;
% num_layers = length(dim_hidden);
% for l = 1:num_layers
%      Y = max(0,W{l}*Y + repmat(b{l}(:),1,size(Y,2)));
% end
% Y = W{end}*Y + repmat(b{end}(:),1,size(Y,2));
% Xout = Y;
% 
% true_min = min(Y)
% true_max = max(Y)


%data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');

% 
% Xin = rect2d(u_min,u_max);
% Xout = net.eval(Xin);
% data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');hold on;

% 
% v1 = W{1}*u + b{1};
% con_hid1 = x11 - alpha*v1(1);
% con_hid2 = -x11 + beta*v1(1);
% con_hid3 = x12 - alpha*v1(2);
% con_hid4 = -x12 + beta*v1(2);
% 
% v2 = W{2}*[x11;x12] + b{2};
% con_hid5 = x21 - alpha*v2(1);
% con_hid6 = -x21 + beta*v2(1);
% con_hid7 = x22 - alpha*v2(2);
% con_hid8 = -x22 + beta*v2(2);
% 
% % Put constraints into structure
% constraints{1} = con_in1; constraints{2} = con_in2;
% constraints{3} = con_hid1; constraints{4} = con_hid2;
% constraints{5} = con_hid3; constraints{6} = con_hid4;
% constraints{7} = con_hid5; constraints{8} = con_hid6;
% constraints{9} = con_hid7; constraints{10} = con_hid8;

