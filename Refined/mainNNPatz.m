clear 
close all
clc

addpath('cvx\sedumi')  
addpath('cvx\sdpt3')  

%% Generate NN parameters
rng('default');

% Input dimension. Can be 1,2,3
dim_in = 1; 

% Input bounds
%u_min = -10*ones(dim_in,1);
%u_max = 10*ones(dim_in,1);
u_min = 0.25*ones(dim_in,1);
u_max = 1.75*ones(dim_in,1);

% Hidden layer dimensions
%dim_hidden = [3,4,3]%[3,5,3];
dim_hidden = [4,4,4,4];

% Ouput dimension. Can be 1,2
dim_out = 1;

% Create NN parameters
dims = [dim_in, dim_hidden, dim_out];
AF = 'sigmoid';
net = nnsequential(dims,AF);
W = net.weights;
b = net.biases;
%save(['net-' num2str(num_hidden_units_per_layer) 'n.mat'],'net');

% Repeated non-linearities on or off
repeated = 1;
% 0: No repeated constraints
% 1: Constraints for all nodes connected to each other
% 2: Constraints for only nodes connected in same hidden layer
% 3: Constraints for only nodes connected in adjacent hidden layers
% 4: Only for relu, enforcing equality constraints, (have realised it won't do anything)
% 5: Only for relu, enforcing equality constraints, (have realised it won't do anything)

% Number of sides of polytope
dim_poly = 8;

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
% 8: Multiply all constraints such that order of the program is quadratic DOES NOT DO ANYTHING

% Controls the order of the polyomials in the SOS program
sos_order = 0;

% Controls the structure of the SOS polynomials
sos_type = 0;
% 0: Full, each multiplier will contain all variables in network
% 1: Vars in constraints, each multiplier will only contain variables in the respective constraint
% 2: Node only, each multiplier will only contain all variables in the respective node
% 3: Layer only, each multiplier will only contain all variables in the respecitive layer
% 4: Layer and neighbours only, each multiplier will only contain all variables in the respecitive layer and neighbouring layers

%load('C:\Users\lascat6145\Documents\ReachSDP-master\ReachSDP-master\nnmpc_nets_di_1.mat'); 
%dim_in = 2; dim_hidden = [15,10]; dim_out = 1; W = weights; b = biases;
%load('C:\Users\lascat6145\Documents\NNPsatz\Refined\sigmoidOptimalValuesResults.mat')

%% SOS 
tic
%k = 1;
%for a = 1:100
if dim_out == 1
    k = 1;
    for c = [-1,1]
        [SOL_bound(k),prog] = NNPatz(net,u_min,u_max,repeated,c,con_type,sos_order,sos_type) %,resultsValues
        NNPsatz_time(k) = prog.solinfo.info.cpusec;
        k = k + 1;
    end
elseif dim_out == 2
    for i = 1:dim_poly
        theta = (i-1)/dim_poly*2*pi;
        C(i,:) = [cos(theta);sin(theta)];
        c = C(i,:)';
        [SOL_bound(i),prog] = NNPatz(net,u_min,u_max,repeated,c,con_type,sos_order,sos_type)
        NNPsatz_time(i) = prog.solinfo.info.cpusec;
        
        %B(i,:) = SOL_bound(i);        
    end
    
    [X_SOS,Y_SOS] = solve_2d_polytope(SOL_bound,dim_poly,C);
    
elseif dim_out >= 3
   disp('Will do higher dimensions later') 
end
%end
NNPsatz_time = sum(NNPsatz_time);
toc

%% Faz method
DeepSDP_bound = [];
if strcmp(AF, 'relu')
% Solver options
options.language = 'yalmip';
options.solver = 'sedumi';
options.verbose = false;
method = 'deepsdp';

%k = 1;
%for a = 1:100
if dim_out == 1
    k = 1;
    for c = [-1,1]
        %repeated = 1; %temp
        [DeepSDP_bound(k),DeepSDP_time(k)] = deep_sdp(net,u_min,u_max,c,repeated,options);
        DeepSDP_bound(k) = DeepSDP_bound(k)*c;
        k = k + 1;
    end
elseif dim_out == 2
    for i = 1:dim_poly
        theta = (i-1)/dim_poly*2*pi;
        C(i,:) = [cos(theta);sin(theta)];
        c = C(i,:)';
        repeated = 0; %temp
        [DeepSDP_bound(i), DeepSDP_time(i),~] = deep_sdp(net,u_min,u_max,c,repeated,options);
        %B(i,:) = bound;
    end
    
    [X_DeepSDP,Y_DeepSDP] = solve_2d_polytope(DeepSDP_bound,dim_poly,C);
    
elseif dim_out >= 3
   disp('Will do higher dimensions later') 
end
%end
DeepSDP_time = sum(DeepSDP_time);
end
%% Evaluate through computation
if dim_in == 1
    Xin  = linspace(u_min,u_max,500000);
elseif dim_in == 2
    Xin = rect2d(u_min,u_max);
elseif dim_in == 3
    [X1,X2,X3] = ndgrid(linspace(u_min(1),u_max(1),50),linspace(u_min(2),u_max(2),50),linspace(u_min(3),u_max(3),50));
    Xin(1,:) = X1(:);
    Xin(2,:) = X2(:);
    Xin(3,:) = X3(:);
elseif dim_in >= 4
    error('Input size too large for now')
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

% Compare outputs
%vpa(SOL_bound,4)
DeepSDP_bound;
tic
[~,~,~,~,IBP_min,IBP_max] = interval_bound_propigation(u_min,u_max,dim_hidden,net)
IBP_time = toc;

% Collate results
if dim_out == 1
    results = zeros(4,3);
    results(1,1) = SOL_bound(1); results(1,2) = SOL_bound(2); results(1,3) = NNPsatz_time; 
    if ~isempty(DeepSDP_bound)
        results(2,1) = DeepSDP_bound(1); results(2,2) = DeepSDP_bound(2); results(2,3) = DeepSDP_time; 
    end
    results(3,1) = IBP_min; results(3,2) = IBP_max; results(3,3) = IBP_time; 
    results(4,1) = true_min; results(4,2) = true_max
    
elseif dim_out == 2

    data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');hold on;
    
    for i=1:dim_poly-1
        plot([X_SOS(i),X_SOS(i+1)],[Y_SOS(i),Y_SOS(i+1)],'red','LineWidth',2);
    end
    h_SOS = plot([X_SOS(dim_poly),X_SOS(1)],[Y_SOS(dim_poly),Y_SOS(1)],'red','LineWidth',2,'DisplayName','NNPsatz');
    
    hold on;
    
    for i=1:dim_poly-1
        plot([X_DeepSDP(i),X_DeepSDP(i+1)],[Y_DeepSDP(i),Y_DeepSDP(i+1)],'black','LineWidth',2);hold on;
    end
    h_DeepSDP = plot([X_DeepSDP(dim_poly),X_DeepSDP(1)],[Y_DeepSDP(dim_poly),Y_DeepSDP(1)],'black','LineWidth',2,'DisplayName','DeepSDP');
    
    % Plot true values
    %plot([IBP_max(1),IBP_max(1)],[IBP_max(1),IBP_max(1)])
    rectangle('Position',[IBP_min(1), IBP_min(2), IBP_max(1) - IBP_min(1), IBP_max(2) - IBP_min(2)],'EdgeColor','m','LineWidth',2)
    
    %legend('True Output','NNPsatz','DeepSDP','IBP')
    
end

% % Collate results
% if dim_out == 1
%     results = zeros(4,3);
%     results(1,1) = vpa(SOL_bound(1),6); results(1,2) = vpa(SOL_bound(2),6); results(1,3) = NNPsatz_time; 
%     if ~isempty(DeepSDP_bound)
%         results(2,1) = DeepSDP_bound(1); results(2,2) = DeepSDP_bound(2); results(2,3) = DeepSDP_time; 
%     end
%     results(3,1) = IBP_min; results(3,2) = IBP_max; results(3,3) = IBP_time; 
%     results(4,1) = true_min; results(4,2) = true_max
% end

%     
%     B = double(B);
%     X_tmp = [];
%     Y_tmp = [];
%     
%     for i=1:dim_poly-1
%         tmp = linsolve(C([i,i+1],:),B([i,i+1],1));
%         X_tmp = [X_tmp;tmp(1)];
%         Y_tmp = [Y_tmp;tmp(2)];
%     end
% 
%     tmp = linsolve(C([1,dim_poly],:),B([1,dim_poly],1));
%     X_tmp = [X_tmp;tmp(1)];
%     Y_tmp = [Y_tmp;tmp(2)];


