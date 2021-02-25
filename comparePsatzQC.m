% Compare the quadratic constraint method in Fazlyab paper with the Psatz
% method.

addpath('cvx\sedumi')  

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

%% Solve using Psatz

% Preprocessing step
[Y_min,Y_max,X_min,X_max,~,~] = net.interval_arithmetic(u_min,u_max);
Ip = find(Y_min>1e-3);
In = find(Y_max<-1e-3);
Ipn = setdiff(1:sum(dim_hidden),union(Ip,In));

% Start off hand crafted for small example, can make general later
syms u x11 x12 x21 x22 y_lim
vars = [u x11 x12 x21 x22]; % y_lim]; 
prog = sosprogram(vars);
prog = sosdecvar(prog,y_lim);

% Input constraints
con_in1 = u - u_min;
con_in2 = u_max - u;
%con_inT1 = con_in1*con_in2;
%[prog,gamma] = sospolyvar(prog,monomials(vars,0:0));

%con_inT1 = con_in1*con_in2;

% Hidden layer constraints
v1 = W{1}*u + b{1};
v2 = W{2}*[x11;x12] + b{2};
orderP = 0;

for j = 1:4
    if any(j == Ip)
        [prog,lambda{j}] = sospolyvar(prog,monomials(vars,0:orderP));
        [prog,eta{j}] = sospolyvar(prog,monomials(vars,0:orderP));
        [prog,nu{j}] = sospolyvar(prog,monomials(vars,0:orderP));
        [prog] = sosineq(prog,nu{j});
        alp(j) = beta;
        bet(j) = beta;
    elseif any(j == In)
        [prog,lambda{j}] = sospolyvar(prog,monomials(vars,0:orderP));
        [prog,eta{j}] = sospolyvar(prog,monomials(vars,0:orderP));
        [prog,nu{j}] = sospolyvar(prog,monomials(vars,0:orderP));
        [prog] = sosineq(prog,eta{j});
        alp(j) = alpha;
        bet(j) = alpha;
    elseif any(j == Ipn)
        [prog,lambda{j}] = sospolyvar(prog,monomials(vars,0:orderP));
        [prog,eta{j}] = sospolyvar(prog,monomials(vars,0:orderP));
        [prog,nu{j}] = sospolyvar(prog,monomials(vars,0:orderP));
        alp(j) = alpha;
        bet(j) = beta;
    end
end

% Output layer constraints
%y_max = -10; % How big the bound is
%c = [cos(0);sin(0)];
%y_lim = 1;
v3 = W{3}*[x21;x22] + b{3};
con_out = (y_lim - v3); % Minus sign can be used to check maximium

% Create Psatz expression
f = -con_out; 

[prog,s1] = sospolyvar(prog,monomials(vars,0:0));
prog = sosineq(prog,s1);
expr = - f - s1*con_in1*con_in2;

x_temp = [x11, x12, x21, x22];
v_temp = [v1(1), v1(2), v2(1), v2(2)];

for j = 1:4
    con_hid{j} = lambda{j}*(x_temp(j) - alp(j)*v_temp(j))*(x_temp(j) - bet(j)*v_temp(j)) + ... 
        nu{j}*(x_temp(j) - bet(j)*v_temp(j)) + eta{j}*(x_temp(j) - alp(j)*v_temp(j));
    expr = expr - con_hid{j};
end

% P-satz refutation condition
prog = sosineq(prog,expr);
prog = sossetobj(prog,y_lim);
solver_opt.solver = 'sedumi';
prog = sossolve(prog,solver_opt);
Psatz_limit = sosgetsol(prog,y_lim)


%% Solve using Fazlyab code

% Solver options
options.language = 'cvx';
options.solver = 'sedumi';
options.verbose = false;
method = 'deepsdp';

% Run DeepSDP
repeated = 0;
%[X_SDP,Y_SDP,chordal_test(i,:)] = output_polytope(net,x_min,x_max,method,repeated,options,m);
c = 1;
[bound, ~,~,~] = deep_sdp(net,u_min,u_max,c,repeated,options);
DSDP_max = bound

c = -1;
[bound, ~,~,~] = deep_sdp(net,u_min,u_max,c,repeated,options);
DSDP_min = -bound

%% Evaluate NN through computation
Xin = linspace(u_min,u_max,50);
Y = Xin;
num_layers = length(dim_hidden);
for l = 1:num_layers
     Y = max(0,W{l}*Y + repmat(b{l}(:),1,size(Y,2)));
end
Y = W{end}*Y + repmat(b{end}(:),1,size(Y,2));
Xout = Y
data = scatter(Xin, Xout);

%% Junk

% 
% 
% con_hid1 = lambda{1}*(x11 - alpha*v1(1))*(x11 - beta*v1(1)) + nu{1}*(x11 - beta*v1(1)) + eta{1}*(x11 - alpha*v1(1));
% con_hid2 = lambda{2}*(x11 - alpha*v1(2))*(x11 - beta*v1(2)) + nu{2}*(x11 - beta*v1(1)) + eta{1}*(x11 - alpha*v1(1));
% con_hid3 = lambda{3}*(x11 - alpha*v1(3))*(x11 - beta*v1(3)) + nu{3}*(x11 - beta*v1(1)) + eta{1}*(x11 - alpha*v1(1));
% con_hid4 = lambda{4}*(x11 - alpha*v1(4))*(x11 - beta*v1(4)) + nu{4}*(x11 - beta*v1(1)) + eta{1}*(x11 - alpha*v1(1));
% % Start off hand crafted for small example, can make general later
% syms u x11 x12 x21 x22 y_lim
% vars = [u x11 x12 x21 x22]; % y_lim]; 
% prog = sosprogram(vars);
% prog = sosdecvar(prog,y_lim);
% 
% % Input constraints
% con_in1 = u - u_min;
% con_in2 = u_max - u;
% 
% % Hidden layer constraints
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
% % Output layer constraints
% %y_max = -10; % How big the bound is
% %c = [cos(0);sin(0)];
% v3 = W{3}*[x21;x22] + b{3};
% con_out = (y_lim - v3); % Minus sign can be used to check maximium
% %con_out = (4.3 - v3)*(v3 - 3.7) % quadratic constraint
% 
% % Put constraints into structure
% constraints{1} = con_in1; constraints{2} = con_in2;
% constraints{3} = con_hid1; constraints{4} = con_hid2;
% constraints{5} = con_hid3; constraints{6} = con_hid4;
% constraints{7} = con_hid5; constraints{8} = con_hid6;
% constraints{9} = con_hid7; constraints{10} = con_hid8;
% 
% % P-sat refutation
% f = -con_out; 
% 
% % Polynomials for cone
% %[prog,s1] = sospolyvar(prog,monomials(vars,0:2));
% %[prog,s2] = sospolyvar(prog,monomials(vars,0:2));
% 
% for j = 1:10
%     [prog,s{j}] = sospolyvar(prog,monomials(vars,0:2));
% end
% 
% % Polynomials for ideal
% % for j = 1:n
% %     [prog,t{j}] = sospolyvar(prog,monomials(vars,0:2));
% % end
% 
% % Create statement 1 + cone + ideal
% %expr = 1 + s{1} + s{2}*f;
% % Rearrange to be in terms of sos, s{1} is sos, s{2} = 1
% expr = - 1 - f;
% 
% for j = 1:10
%     expr = expr + s{j}*constraints{j};
% end
% 
% % Enforce SOS on polynomials s_j
% for j = 1:10
%     prog = sosineq(prog,s{j});
% end
% 
% % P-satz refutation condition
% prog = sosineq(prog,expr);
% prog = sossetobj(prog,-y_lim);
% solver_opt.solver = 'sedumi';
% prog = sossolve(prog,solver_opt);
% Psatz_limit = sosgetsol(prog,y_lim)


% % Hidden layer constraints
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

% % Output layer constraints
% %y_max = -10; % How big the bound is
% %c = [cos(0);sin(0)];
% v3 = W{3}*[x21;x22] + b{3};
% con_out = (y_lim - v3); % Minus sign can be used to check maximium
% %con_out = (4.3 - v3)*(v3 - 3.7) % quadratic constraint
% 
% % Put constraints into structure
% constraints{1} = con_in1; constraints{2} = con_in2;
% constraints{3} = con_hid1; constraints{4} = con_hid2;
% constraints{5} = con_hid3; constraints{6} = con_hid4;
% constraints{7} = con_hid5; constraints{8} = con_hid6;
% constraints{9} = con_hid7; constraints{10} = con_hid8;
% 
% % P-sat refutation
% f = con_out; 
% 
% % Polynomials for cone
% %[prog,s1] = sospolyvar(prog,monomials(vars,0:2));
% %[prog,s2] = sospolyvar(prog,monomials(vars,0:2));
% 
% % For terms g_j
% for j = 1:10
%     [prog,s{j}] = sospolyvar(prog,monomials(vars,0:2));
% end
% 
% % For terms g_j*g_k
% for j = 1:10
%     for k = 1:10
%         if j ~= k
%             [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0:2));
%         end
%     end
% end
% 
% % Polynomials for ideal
% % for j = 1:n
% %     [prog,t{j}] = sospolyvar(prog,monomials(vars,0:2));
% % end
% 
% % Create statement 1 + cone + ideal
% %expr = 1 + s{1} + s{2}*f;
% % Rearrange to be in terms of sos, s{1} is sos, s{2} = 1
% expr = - 1 - f;
% 
% for j = 1:10
%     expr = expr + s{j}*constraints{j};
% end
% 
% for j = 1:10
%     for k = 1:10
%         if j ~= k
%             expr = expr + s2{j,k}*constraints{j}*constraints{k};
%         end
%     end
% end
% 
% % Enforce SOS on polynomials s_j
% for j = 1:10
%     prog = sosineq(prog,s{j});
% end
% 
% for j = 1:10
%     for k = 1:10
%         if j ~= k
%             prog = sosineq(prog,s2{j,k});
%         end
%     end
% end


