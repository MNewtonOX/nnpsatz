clear 
close all
clc

addpath('cvx\sedumi')  
addpath('cvx\sdpt3')  

%% Generate NN parameters
rng('default');

% Input dimension. Value can be 1, 2 or 3
dim_in = 1; 

% Input bounds (hyper-rectangle constraint)
u_min = -10*ones(dim_in,1);
u_max = 10*ones(dim_in,1);

% Hidden layer dimensions
dim_hidden = [2,2];

% Ouput dimension. Calue can be 1 or 2
dim_out = 1;

% Activation function type. Can be relu, sigmoid or tanh
AF = 'relu'; 

% Create NN parameters
dims = [dim_in, dim_hidden, dim_out];
net = NNsetup(dims,AF);

% Repeated non-linearities on or off
repeated = 0;
% 0: No repeated constraints
% 1: Constraints for all nodes connected to each other
% 2: Constraints for only nodes connected in same hidden layer
% 3: Constraints for only nodes connected in adjacent hidden layers
% 4: Only for relu, enforcing equality constraints, (have realised this won't do anything)
% 5: Only for relu, enforcing equality constraints, (have realised this won't do anything)

% Number of sides of polytope (for output dimension 2 only).
dim_poly = 6;

% Controls how the constraints are multiplied together
con_type = 0;
% 0: No overlapping constraints.
% 1: Multiply all constraints by one other constraint, excluding input constraints
% 2: Multiply all constraints by one other constraint
% 3: Multiply all constraints within the same node
% 4: Multiply all constraints within the same layer
% 5: Multiply all constraints within the same layer and neighbouring layers
% 6: Multiply all constraints within the same layer and neighbouring layers, excluding input constraints
% 7: Multiply all constraints within the neighbouring layers, excluding input constraints
% 8: Multiply all constraints such that order of the program is quadratic

% Controls the order of the polyomials in the SOS program
sos_order = 0;

% Controls the structure of the SOS polynomials, (if sos_order = 0, this parameter won't do anything)
sos_type = 0;
% 0: Full, each multiplier will contain all variables in network 
% 1: Vars in constraints, each multiplier will only contain variables in the respective constraint
% 2: Node only, each multiplier will only contain all variables in the respective node
% 3: Layer only, each multiplier will only contain all variables in the respecitive layer
% 4: Layer and neighbours only, each multiplier will only contain all variables in the respecitive layer and neighbouring layers

%% NNPsatz 
solver_opt.solver = 'sdpt3';
if dim_out == 1
    SOL_bound = zeros(1,2);
    NNPsatz_time = zeros(1,2);
    c = [-1,1];
    for i = 1:2 
        [SOL_bound(i),prog] = NNPsatz(net,u_min,u_max,repeated,c(i),con_type,sos_order,sos_type,solver_opt.solver);
        NNPsatz_time(i) = prog.solinfo.info.cpusec;
    end
elseif dim_out == 2
    SOL_bound = zeros(1,dim_poly);
    NNPsatz_time = zeros(1,dim_poly);
    C = zeros(dim_poly,2);
    for i = 1:dim_poly
        theta = (i-1)/dim_poly*2*pi;
        C(i,:) = [cos(theta);sin(theta)];
        c = C(i,:)';
        [SOL_bound(i),prog] = NNPsatz(net,u_min,u_max,repeated,c,con_type,sos_order,sos_type,solver_opt.solver);
        NNPsatz_time(i) = prog.solinfo.info.cpusec; 
    end
    
    [X_SOS,Y_SOS] = solvePolytope(SOL_bound,dim_poly,C);
    
elseif dim_out >= 3
   disp('Higher dimension will be added later') 
end

NNPsatz_time = sum(NNPsatz_time);

%% DeepSDP method (Mahyar Fazlyab, Manfred Morari, George J Pappas, June 2020)
DeepSDP_bound = [];
if strcmp(AF, 'relu')
    
% Solver options
options.language = 'yalmip';
options.solver = 'sedumi';
options.verbose = true;
method = 'deepsdp';

if dim_out == 1
    DeepSDP_bound = zeros(1,2);
    DeepSDP_time = zeros(1,2);
    c = [-1,1];
    for i = 1:2 
        [DeepSDP_bound(i),DeepSDP_time(i)] = deep_sdp(net,u_min,u_max,c(i),repeated,options);
        DeepSDP_bound(i) = DeepSDP_bound(i)*c(i);
    end
elseif dim_out == 2
    DeepSDP_bound = zeros(1,dim_poly);
    DeepSDP_time = zeros(1,dim_poly);
    C = zeros(dim_poly,2);
    for i = 1:dim_poly
        theta = (i-1)/dim_poly*2*pi;
        C(i,:) = [cos(theta);sin(theta)];
        c = C(i,:)';
        [DeepSDP_bound(i), DeepSDP_time(i),~] = deep_sdp(net,u_min,u_max,c,repeated,options);
    end
    
    [X_DeepSDP,Y_DeepSDP] = solvePolytope(DeepSDP_bound,dim_poly,C);
    
elseif dim_out >= 3
   disp('Will do higher dimensions later') 
end
DeepSDP_time = sum(DeepSDP_time);

end
%% Evaluate bounds through computation
if dim_in == 1
    Xin  = linspace(u_min,u_max,500000);
elseif dim_in == 2
    Xin = grid2D(u_min,u_max);
elseif dim_in == 3
    [X1,X2,X3] = ndgrid(linspace(u_min(1),u_max(1),50),linspace(u_min(2),u_max(2),50),linspace(u_min(3),u_max(3),50));
    Xin(1,:) = X1(:);
    Xin(2,:) = X2(:);
    Xin(3,:) = X3(:);
elseif dim_in >= 4
    error('Input dimension too big, higher dimensions will be added later')
end

Xout = net.eval(Xin);
if dim_out == 1
    true_max = max(Xout);
    true_min = min(Xout);
elseif dim_out == 2
    true_max_x = max(Xout(1,:));
    true_min_x = min(Xout(1,:));
    true_max_y = max(Xout(2,:));
    true_min_y = min(Xout(2,:));
end

%% Compare bounds
tic
[~,~,~,~,IBP_min,IBP_max] = intervalBoundPropagation(u_min,u_max,dim_hidden,net);
IBP_time = toc;

% Collate results
if dim_out == 1
    results = zeros(4,3);
    results(1,1) = SOL_bound(1); results(1,2) = SOL_bound(2); results(1,3) = NNPsatz_time; 
    if ~isempty(DeepSDP_bound)
        results(2,1) = DeepSDP_bound(1); results(2,2) = DeepSDP_bound(2); results(2,3) = DeepSDP_time; 
    end
    results(3,1) = IBP_min; results(3,2) = IBP_max; results(3,3) = IBP_time; 
    results(4,1) = true_min; results(4,2) = true_max;
    results
elseif dim_out == 2    
    plotResults(dim_poly,Xout,X_SOS,Y_SOS,X_DeepSDP,Y_DeepSDP)
end

