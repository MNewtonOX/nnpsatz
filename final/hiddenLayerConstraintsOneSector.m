% Function for hidden layer constraints
function [eq_constraints, ineq_constraints] = hiddenLayerConstraintsOneSector(net,u_min,u_max,u,x)

% Extract weights and biases
W = net.weights;
b = net.biases;

% Extract dimensions
dims = net.dims;
dim_hidden = dims(2:end-1);

% Extract activation function
AF = net.activation;

% Create cell for constraints
ineq_constraints = {};
eq_constraints = {};

icount = 1; %inequality
ecount = 1; %equality

%% ReLU
if strcmp(AF, 'relu')

alpha = 0;
beta = 1;

% Pre-processing step to find active and inactive neurons
[Y_min,Y_max,X_min,X_max] = intervalBoundPropagation(u_min,u_max,dim_hidden,net);
Ip = find(Y_min>1e-3);
In = find(Y_max<-1e-3);
Ipn = setdiff(1:sum(dim_hidden),union(Ip,In));

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
        % Sector constraints
        if any(node_num == Ip) 
            ineq_constraints{icount,1} = x_curr_layer(k) - alpha*v{j}(k); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
            eq_constraints{ecount,1} = x_curr_layer(k) - beta*v{j}(k); eq_constraints{ecount,2} = [j,k]; ecount = ecount + 1;
            eq_constraints{ecount,1} = (x_curr_layer(k) - beta*v{j}(k))*(x_curr_layer(k) - beta*v{j}(k)); eq_constraints{ecount,2} = [j,k]; ecount = ecount + 1;
        elseif any(node_num == In) 
            eq_constraints{ecount,1} = x_curr_layer(k) - alpha*v{j}(k); eq_constraints{ecount,2} = [j,k]; ecount = ecount + 1;
            ineq_constraints{icount,1} = x_curr_layer(k) - beta*v{j}(k); ineq_constraints{icount,2} = [j,k]; icount = icount + 1;
            eq_constraints{ecount,1} = (x_curr_layer(k) - alpha*v{j}(k))*(x_curr_layer(k) - alpha*v{j}(k)); eq_constraints{ecount,2} = [j,k]; ecount = ecount + 1;
        elseif any(node_num == Ipn) 
            ineq_constraints{icount,1} = x_curr_layer(k) - alpha*v{j}(k); ineq_constraints{icount,2} = [j,k]; icount = icount + 1;
            ineq_constraints{icount,1} = x_curr_layer(k) - beta*v{j}(k); ineq_constraints{icount,2} = [j,k]; icount = icount + 1;
            eq_constraints{ecount,1} = (x_curr_layer(k) - alpha*v{j}(k))*(x_curr_layer(k) - beta*v{j}(k)); eq_constraints{ecount,2} = [j,k]; ecount = ecount + 1;
        end
        
        % Interval bounds
        ineq_constraints{icount,1} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1;
        
    end
end

end

%% Sigmoid
if strcmp(AF, 'sigmoid')
    
% Pre-procesing step to obtain approximate bounds    
[~,~,X_min,X_max] = intervalBoundPropagation(u_min,u_max,dim_hidden,net);

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
        
        % Simple sector
        ineq_constraints{icount,1} = (x_curr_layer(k) - 0.5)*((0.25*v{j}(k) + 0.5) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
 
        % Pre-processing bounds
        ineq_constraints{icount,1} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
    end
end
end

%% tanh
if strcmp(AF, 'tanh')
    
% Pre-procesing step to obtain approximate bounds    
[~,~,X_min,X_max] = intervalBoundPropagation(u_min,u_max,dim_hidden,net);

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
        
        % Simple sector
        ineq_constraints{icount,1} = (x_curr_layer(k))*(v{j}(k) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
 
        % Pre-processing bounds
        ineq_constraints{icount,1} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
    end
end
end

%% Functions 
function [y] = sig(x)
    y = 1./(1+exp(-x));
end

end