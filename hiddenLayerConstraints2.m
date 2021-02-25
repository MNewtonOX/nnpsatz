% Function for hidden layer constraints
function [eq_constraints, ineq_constraints] = hiddenLayerConstraints(net,AF,alpha,beta,u_min,u_max,u,x,y,dim_in,dim_hidden,dim_out,repeated)

W = net.weights;
b = net.biases;

% Create cell for constraints
ineq_constraints = {};
eq_constraints = {};
icount = 1; %inequality
ecount = 1; %equality

if strcmp(AF, 'relu')
    
% Pre-processing step to find active and inactive neurons
[Y_min2,Y_max2,X_min2,X_max2,~,~] = net.interval_arithmetic(u_min,u_max);
[Y_min,Y_max,X_min,X_max] = interval_bound_propigation(u_min,u_max,dim_hidden,net);
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
            ineq_constraints{icount} = x_curr_layer(k) - alpha*v{j}(k); icount = icount + 1;
            eq_constraints{ecount} = x_curr_layer(k) - beta*v{j}(k); ecount = ecount + 1;
            eq_constraints{ecount} = (x_curr_layer(k) - beta*v{j}(k))*(x_curr_layer(k) - beta*v{j}(k)); ecount = ecount + 1;
        elseif any(node_num == In) 
            eq_constraints{ecount} = x_curr_layer(k) - alpha*v{j}(k); ecount = ecount + 1;
            ineq_constraints{icount} = x_curr_layer(k) - beta*v{j}(k); icount = icount + 1;
            eq_constraints{ecount} = (x_curr_layer(k) - alpha*v{j}(k))*(x_curr_layer(k) - alpha*v{j}(k)); ecount = ecount + 1;
        elseif any(node_num == Ipn) 
            ineq_constraints{icount} = x_curr_layer(k) - alpha*v{j}(k); icount = icount + 1;
            ineq_constraints{icount} = x_curr_layer(k) - beta*v{j}(k); icount = icount + 1;
            eq_constraints{ecount} = (x_curr_layer(k) - alpha*v{j}(k))*(x_curr_layer(k) - beta*v{j}(k)); ecount = ecount + 1;
        end
        
        % Interval bounds
        %ineq_constraints{icount} = x_curr_layer(k) - X_min_curr_layer(k); icount = icount + 1;
        %ineq_constraints{icount} = -x_curr_layer(k) + X_max_curr_layer(k); icount = icount + 1;
        ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;
        
    end
end

end

% Slope restricted constraints (same for relu, sigmoid and tanh so have put outside if statement)
if repeated == 1   
    % Get pre-activation variables
    for j = 1:length(dim_hidden)
        if j == 1
            x_prev_layer = u;
            x_curr_layer = x(1:dim_hidden(j));
            v_temp{j} = W{j}*u + b{j};
        else
            x_prev_layer = x(sum(dim_hidden(1:j-2)) + 1 : sum(dim_hidden(1:j-1)));
            x_curr_layer = x(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
            v_temp{j} = W{j}*x_prev_layer + b{j};
        end
    end
    v_temp = cat(1,v_temp{:});
    
    for j = 1:sum(dim_hidden)
        for k = j+1:sum(dim_hidden)
            x1 = x(j); v1 = v_temp(j);
            x2 = x(k); v2 = v_temp(k);
            %ineq_constraints{icount} = (x2 - x1)*(v2 - v1) - 0*(v2 - v1)^2; icount = icount + 1;
            %ineq_constraints{icount} = 1*(v2 - v1)^2 - (x2 - x1)*(v2 - v1); icount = icount + 1;
            ineq_constraints{icount} = ((x2 - x1) - 0*(v2 - v1))*(1*(v2 - v1) - (x2 - x1)); icount = icount + 1;
        end
    end
end


if strcmp(AF, 'sigmoid')
[Y_min,Y_max,X_min,X_max] = interval_bound_propigation(u_min,u_max,dim_hidden,net);
alpha = 0;
beta = 1;
for j = 1:length(dim_hidden)
    if j == 1
        x_prev_layer = u;
        x_curr_layer = x(1:dim_hidden(j));
        v{j} = W{j}*u + b{j};
        X_min_curr_layer = X_min(1:dim_hidden(j));
        X_max_curr_layer = X_max(1:dim_hidden(j));
        Y_min_curr_layer = Y_min(1:dim_hidden(j));
        Y_max_curr_layer = Y_max(1:dim_hidden(j));
    else
        x_prev_layer = x(sum(dim_hidden(1:j-2)) + 1 : sum(dim_hidden(1:j-1)));
        x_curr_layer = x(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        v{j} = W{j}*x_prev_layer + b{j};
        X_min_curr_layer = X_min(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        X_max_curr_layer = X_max(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        Y_min_curr_layer = Y_min(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        Y_max_curr_layer = Y_max(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
    end
    for k = 1:dim_hidden(j)
        % Function limits
        ineq_constraints{icount} = x_curr_layer(k) - alpha; icount = icount + 1;
        ineq_constraints{icount} = -x_curr_layer(k) + beta; icount = icount + 1;
        %ineq_constraints{icount} = (x_curr_layer(k) - alpha)*(-x_curr_layer(k) + beta); icount = icount + 1;
        
        % Sector constraints
        ineq_constraints{icount} = (x_curr_layer(k) - 0.5)*(0.25*v{j}(k) + 0.5 - x_curr_layer(k)); icount = icount + 1;
        
        % Under linear constraints
        if Y_min_curr_layer(k) > 0
            grad1 = (X_max_curr_layer(k) - X_min_curr_layer(k))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
            c1 = X_min_curr_layer(k) - grad1*Y_min_curr_layer(k);
            ineq_constraints{icount} = -(grad1*v{j}(k) + c1) + x_curr_layer(k); icount = icount + 1;
        elseif Y_max_curr_layer(k) < 0
            grad1 = (X_max_curr_layer(k) - X_min_curr_layer(k))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
            c1 = X_min_curr_layer(k) - grad1*Y_min_curr_layer(k);
            ineq_constraints{icount} = grad1*v{j}(k) + c1 - x_curr_layer(k); icount = icount + 1;
        else
            if abs(Y_max_curr_layer(k)) > abs(Y_min_curr_layer(k))
                grad1 = (X_max_curr_layer(k) - 0.5)/(Y_max_curr_layer(k) - 0);
                ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + 0.5))*(0.25*v{j}(k) + 0.5 - x_curr_layer(k)); icount = icount + 1;
            else
                grad1 = (X_min_curr_layer(k) - 0.5)/(Y_min_curr_layer(k) - 0);
                ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + 0.5))*(0.25*v{j}(k) + 0.5 - x_curr_layer(k)); icount = icount + 1;
            end
%             grad1 = (X_max_curr_layer(k) - 0.5)/(Y_max_curr_layer(k) - 0);
%             ineq_constraints{icount} = x_curr_layer(k) - grad1*v{j}(k); icount = icount + 1;
%             grad2 = -X_min_curr_layer(k)/Y_min_curr_layer(k);
%             ineq_constraints{icount} = -x_curr_layer(k) + grad2*v{j}(k); icount = icount + 1;
        end
        
        % Derivative constraints
        %ineq_constraints{icount} = x_curr_layer(k)*(1 - x_curr_layer(k)); icount = icount + 1;
        %ineq_constraints{icount} = 0.25 - x_curr_layer(k)*(1 - x_curr_layer(k)); icount = icount + 1;
        
        % Find lower bound line
        %gradl = (difsig(Y_max_curr_layer(k)) - difsig(Y_min_curr_layer(k)))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
        %cl = difsig(Y_min_curr_layer(k)) - gradl*Y_min_curr_layer(k);
        
        %ineq_constraints{icount} = x_curr_layer(k)*(1 - x_curr_layer(k)) - (gradl*v{j}(k) + cl) ; icount = icount + 1;
        
%         % Second derivative constraints
%         second_diff = x_curr_layer(k)*(1 - x_curr_layer(k))*(1 - 2*x_curr_layer(k));
%         ineq_constraints{icount} = second_diff + 0.1; icount = icount + 1;
%         ineq_constraints{icount} = -second_diff + 0.1; icount = icount + 1;
        
        % Pre-processing bounds
        ineq_constraints{icount} = (-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;
        ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k)); icount = icount + 1;
        ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;
    end
end
end

if strcmp(AF, 'tanh')
%[Y_min,Y_max,X_min,X_max,~,~] = net.interval_arithmetic(u_min,u_max);
[Y_min,Y_max,X_min,X_max] = interval_bound_propigation(u_min,u_max,dim_hidden,net);
for j = 1:length(dim_hidden)
    if j == 1
        x_prev_layer = u;
        x_curr_layer = x(1:dim_hidden(j));
        v{j} = W{j}*u + b{j};
        X_min_curr_layer = X_min(1:dim_hidden(j));
        X_max_curr_layer = X_max(1:dim_hidden(j));
        Y_min_curr_layer = Y_min(1:dim_hidden(j));
        Y_max_curr_layer = Y_max(1:dim_hidden(j));
    else
        x_prev_layer = x(sum(dim_hidden(1:j-2)) + 1 : sum(dim_hidden(1:j-1)));
        x_curr_layer = x(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        v{j} = W{j}*x_prev_layer + b{j};
        X_min_curr_layer = X_min(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        X_max_curr_layer = X_max(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        Y_min_curr_layer = Y_min(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        Y_max_curr_layer = Y_max(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
    end
    for k = 1:dim_hidden(j)
        % Function limits
        ineq_constraints{icount} = x_curr_layer(k) + 1; icount = icount + 1;
        ineq_constraints{icount} = -x_curr_layer(k) + 1; icount = icount + 1;
        ineq_constraints{icount} = (x_curr_layer(k) + 1)*(-x_curr_layer(k) + 1); icount = icount + 1;
        
        % Sector constraints
        ineq_constraints{icount} = (x_curr_layer(k) - 0)*(v{j}(k) - x_curr_layer(k)); icount = icount + 1;
        
        % Derivative constraints
        %ineq_constraints{icount} = x_curr_layer(k)^2; icount = icount + 1;
        %ineq_constraints{icount} = 1 - x_curr_layer(k)^2; icount = icount + 1;
        
        % Find lower bound line
        gradl = (diftanh(Y_max_curr_layer(k)) - diftanh(Y_min_curr_layer(k)))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
        cl = diftanh(Y_min_curr_layer(k)) - gradl*Y_min_curr_layer(k);
        
        ineq_constraints{icount} = 1 - x_curr_layer(k)^2 - (gradl*v{j}(k) + cl) ; icount = icount + 1;
        
%         % Second derivative constraints
%         second_diff = 2*x_curr_layer(k)*(x_curr_layer(k)^2 - 1);
%         ineq_constraints{icount} = second_diff + (4*3^(1/2))/9; icount = icount + 1;
%         ineq_constraints{icount} = -second_diff + (4*3^(1/2))/9; icount = icount + 1;
        
        % Pre-processing bounds
        ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;
    end
end
end

% Functions for derivatives
function [y] = difsig(x)
    y = sig(x)*(1 - sig(x));
end

function [y] = sig(x)
    y = 1./(1+exp(-x));
end

function [y] = diftanh(x)
    y = 1 - tanh(x)^2;
end

end