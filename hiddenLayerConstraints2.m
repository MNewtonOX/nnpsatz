% Function for hidden layer constraints
function [eq_constraints, ineq_constraints] = hiddenLayerConstraints(net,AF,alpha,beta,u_min,u_max,u,x,y,dim_in,dim_hidden,dim_out,repeated)

W = net.weights;
b = net.biases;

% Create cell for constraints
ineq_constraints = {};
eq_constraints = {};
icount = 1; %inequality
ecount = 1; %equality

%% ReLU
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

%% Sigmoid
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
        %ineq_constraints{icount} = x_curr_layer(k) - alpha; icount = icount + 1;
        %ineq_constraints{icount} = -x_curr_layer(k) + beta; icount = icount + 1;
        %ineq_constraints{icount} = (x_curr_layer(k) - alpha)*(-x_curr_layer(k) + beta); icount = icount + 1;
        
        % Redundant as stated in next iteration I think
        %ineq_constraints{icount} = v{j}(k) - Y_min_curr_layer(k); icount = icount + 1;
        %ineq_constraints{icount} = -v{j}(k) + Y_max_curr_layer(k); icount = icount + 1;
        
        % Sector constraints
        %ineq_constraints{icount} = (x_curr_layer(k) - 0.5)*(0.25*v{j}(k) + 0.5 - x_curr_layer(k)); icount = icount + 1;
        
        % Ofset sector constraints
        if Y_min_curr_layer(k) > 0
            Xmid = (X_max_curr_layer(k) - X_min_curr_layer(k))/2;
            Ymid = (Y_max_curr_layer(k) - Y_min_curr_layer(k))/2;
            grad1 = (X_min_curr_layer(k) - Xmid)/(Y_min_curr_layer(k) - Ymid); 
            c1 = X_min_curr_layer(k) - grad1*Y_min_curr_layer(k);
            grad2 = (X_max_curr_layer(k) - Xmid)/(Y_max_curr_layer(k) - Ymid);
            c2 = X_max_curr_layer(k) - grad2*Y_max_curr_layer(k);
            ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + c1))*(-x_curr_layer(k) + grad2*v{j}(k) + c2); icount = icount + 1;
        elseif Y_max_curr_layer(k) < 0
            Xmid = (X_max_curr_layer(k) - X_min_curr_layer(k))/2;
            Ymid = (Y_max_curr_layer(k) - Y_min_curr_layer(k))/2;
            grad1 = (X_min_curr_layer(k) - Xmid)/(Y_min_curr_layer(k) - Ymid); 
            c1 = X_min_curr_layer(k) - grad1*Y_min_curr_layer(k);
            grad2 = (X_max_curr_layer(k) - Xmid)/(Y_max_curr_layer(k) - Ymid);
            c2 = X_max_curr_layer(k) - grad2*Y_max_curr_layer(k);
            ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + c1))*(-x_curr_layer(k) + grad2*v{j}(k) + c2); icount = icount + 1;
        else
            if abs(Y_max_curr_layer(k)) > abs(Y_min_curr_layer(k))
                grad1 = (X_max_curr_layer(k) - 0.5)/(Y_max_curr_layer(k) - 0);
                ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + 0.5))*(0.25*v{j}(k) + 0.5 - x_curr_layer(k)); icount = icount + 1;
            else
                grad1 = (X_min_curr_layer(k) - 0.5)/(Y_min_curr_layer(k) - 0);
                ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + 0.5))*(0.25*v{j}(k) + 0.5 - x_curr_layer(k)); icount = icount + 1;
            end
            
        end
        
% %        Under linear constraints
%         if Y_min_curr_layer(k) > 0
%             grad1 = (X_max_curr_layer(k) - X_min_curr_layer(k))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
%             c1 = X_min_curr_layer(k) - grad1*Y_min_curr_layer(k);
%             ineq_constraints{icount} = -(grad1*v{j}(k) + c1) + x_curr_layer(k); icount = icount + 1;
%         elseif Y_max_curr_layer(k) < 0
%             grad1 = (X_max_curr_layer(k) - X_min_curr_layer(k))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
%             c1 = X_min_curr_layer(k) - grad1*Y_min_curr_layer(k);
%             ineq_constraints{icount} = grad1*v{j}(k) + c1 - x_curr_layer(k); icount = icount + 1;
%         else
%             if abs(Y_max_curr_layer(k)) > abs(Y_min_curr_layer(k))
%                 grad1 = (X_max_curr_layer(k) - 0.5)/(Y_max_curr_layer(k) - 0);
%                 ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + 0.5))*(0.25*v{j}(k) + 0.5 - x_curr_layer(k)); icount = icount + 1;
%             else
%                 grad1 = (X_min_curr_layer(k) - 0.5)/(Y_min_curr_layer(k) - 0);
%                 ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + 0.5))*(0.25*v{j}(k) + 0.5 - x_curr_layer(k)); icount = icount + 1;
%             end
% %             grad1 = (X_max_curr_layer(k) - 0.5)/(Y_max_curr_layer(k) - 0);
% %             ineq_constraints{icount} = x_curr_layer(k) - grad1*v{j}(k); icount = icount + 1;
% %             grad2 = -X_min_curr_layer(k)/Y_min_curr_layer(k);
% %             ineq_constraints{icount} = -x_curr_layer(k) + grad2*v{j}(k); icount = icount + 1;
%         end
        
        % Derivative constraints
        %ineq_constraints{icount} = x_curr_layer(k)*(1 - x_curr_layer(k)); icount = icount + 1;
        %ineq_constraints{icount} = 0.25 - x_curr_layer(k)*(1 - x_curr_layer(k)); icount = icount + 1;
        
% %        Find lower bound line
%         gradl = (difsig(Y_max_curr_layer(k)) - difsig(Y_min_curr_layer(k)))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
%         cl = difsig(Y_min_curr_layer(k)) - gradl*Y_min_curr_layer(k);
%         
%         ineq_constraints{icount} = x_curr_layer(k)*(1 - x_curr_layer(k)) - (gradl*v{j}(k) + cl) ; icount = icount + 1;
        
%         % Second derivative constraints
%         second_diff = x_curr_layer(k)*(1 - x_curr_layer(k))*(1 - 2*x_curr_layer(k));
%         ineq_constraints{icount} = second_diff + 0.1; icount = icount + 1;
%         ineq_constraints{icount} = -second_diff + 0.1; icount = icount + 1;
        
        % Pre-processing bounds
%        ineq_constraints{icount} = (-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;
%        ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k)); icount = icount + 1;
        ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;
    end
end
end

%% tanh
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
        %ineq_constraints{icount} = x_curr_layer(k) + 1; icount = icount + 1;
        %ineq_constraints{icount} = -x_curr_layer(k) + 1; icount = icount + 1;
        %ineq_constraints{icount} = (x_curr_layer(k) + 1)*(-x_curr_layer(k) + 1); icount = icount + 1;
        
        % Sector constraints
        %ineq_constraints{icount} = (x_curr_layer(k) - 0)*(v{j}(k) - x_curr_layer(k)); icount = icount + 1;
        
        % Ofset sector constraints
        if Y_min_curr_layer(k) > 0
            Xmid = (X_max_curr_layer(k) - X_min_curr_layer(k))/2;
            Ymid = (Y_max_curr_layer(k) - Y_min_curr_layer(k))/2;
            grad1 = (X_min_curr_layer(k) - Xmid)/(Y_min_curr_layer(k) - Ymid); 
            c1 = X_min_curr_layer(k) - grad1*Y_min_curr_layer(k);
            grad2 = (X_max_curr_layer(k) - Xmid)/(Y_max_curr_layer(k) - Ymid);
            c2 = X_max_curr_layer(k) - grad2*Y_max_curr_layer(k);
            ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + c1))*(-x_curr_layer(k) + grad2*v{j}(k) + c2); icount = icount + 1;
        elseif Y_max_curr_layer(k) < 0
            Xmid = (X_max_curr_layer(k) - X_min_curr_layer(k))/2;
            Ymid = (Y_max_curr_layer(k) - Y_min_curr_layer(k))/2;
            grad1 = (X_min_curr_layer(k) - Xmid)/(Y_min_curr_layer(k) - Ymid); 
            c1 = X_min_curr_layer(k) - grad1*Y_min_curr_layer(k);
            grad2 = (X_max_curr_layer(k) - Xmid)/(Y_max_curr_layer(k) - Ymid);
            c2 = X_max_curr_layer(k) - grad2*Y_max_curr_layer(k);
            ineq_constraints{icount} = (x_curr_layer(k) - (grad1*v{j}(k) + c1))*(-x_curr_layer(k) + grad2*v{j}(k) + c2); icount = icount + 1;
        else
            if abs(Y_max_curr_layer(k)) > abs(Y_min_curr_layer(k))
                grad1 = (X_max_curr_layer(k) - 0)/(Y_max_curr_layer(k) - 0);
                ineq_constraints{icount} = (x_curr_layer(k) - grad1*v{j}(k))*(1*v{j}(k) - x_curr_layer(k)); icount = icount + 1;
            else
                grad1 = (X_min_curr_layer(k) - 0)/(Y_min_curr_layer(k) - 0);
                ineq_constraints{icount} = (x_curr_layer(k) - grad1*v{j}(k))*(1*v{j}(k) - x_curr_layer(k)); icount = icount + 1;
            end
            
        end
        
        % Derivative constraints
        %ineq_constraints{icount} = x_curr_layer(k)^2; icount = icount + 1;
        %ineq_constraints{icount} = 1 - x_curr_layer(k)^2; icount = icount + 1;
        
%         % Find lower bound line
%         gradl = (diftanh(Y_max_curr_layer(k)) - diftanh(Y_min_curr_layer(k)))/(Y_max_curr_layer(k) - Y_min_curr_layer(k));
%         cl = diftanh(Y_min_curr_layer(k)) - gradl*Y_min_curr_layer(k);
%         
%         ineq_constraints{icount} = 1 - x_curr_layer(k)^2 - (gradl*v{j}(k) + cl) ; icount = icount + 1;
        
%         % Second derivative constraints
%         second_diff = 2*x_curr_layer(k)*(x_curr_layer(k)^2 - 1);
%         ineq_constraints{icount} = second_diff + (4*3^(1/2))/9; icount = icount + 1;
%         ineq_constraints{icount} = -second_diff + (4*3^(1/2))/9; icount = icount + 1;
        
        % Pre-processing bounds
        ineq_constraints{icount} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); icount = icount + 1;
    end
end
end

%% Slope constraints
% Slope restricted constraints (same for relu, sigmoid and tanh so have put outside if statement)

% All nodes connected to each other
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

% Only nodes connected in same hidden layer
if repeated == 2
for j = 1:length(dim_hidden)
    if j == 1
        x_prev_layer = u;
        x_curr_layer = x(1:dim_hidden(j));
        v{j} = W{j}*u + b{j};
    else
        x_prev_layer = x(sum(dim_hidden(1:j-2)) + 1 : sum(dim_hidden(1:j-1)));
        x_curr_layer = x(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        v{j} = W{j}*x_prev_layer + b{j};
    end
    for k = 1:dim_hidden(j)
        for m = k:dim_hidden(j)
            ineq_constraints{icount} = -(x_curr_layer(k) - x_curr_layer(m) - 0)*(x_curr_layer(k) - x_curr_layer(m) - 1*(v{j}(k) - v{j}(m))); icount = icount + 1;  
        end
    end
end    
end

% Only nodes in adjacent hidden layers
if repeated == 3
for j = 1:length(dim_hidden)
    if j == 1
        x_prev_layer = u;
        x_curr_layer = x(1:dim_hidden(j));
        v{j} = W{j}*u + b{j};
    else
        x_prev_layer = x(sum(dim_hidden(1:j-2)) + 1 : sum(dim_hidden(1:j-1)));
        x_curr_layer = x(sum(dim_hidden(1:j-1)) + 1 : sum(dim_hidden(1:j)));
        v{j} = W{j}*x_prev_layer + b{j};
    end
    for k = 1:dim_hidden(j)
        for m = k:dim_hidden(j)
            ineq_constraints{icount} = -(x_curr_layer(k) - x_curr_layer(m) - 0)*(x_curr_layer(k) - x_curr_layer(m) - 1*(v{j}(k) - v{j}(m))); icount = icount + 1;  
        end
    end
    if j > 1
        for k = 1:dim_hidden(j)
            for m = 1:dim_hidden(j-1)
                ineq_constraints{icount} = -(x_curr_layer(k) - x_prev_layer(m) - 0)*(x_curr_layer(k) - x_prev_layer(m) - 1*(v{j}(k) - v{j-1}(m))); icount = icount + 1;
            end
        end
    end
end    
end

% Only for relu, enforcing equality constraints, have realised it won't do
% anything
if repeated == 4 && strcmp(AF, 'relu')
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
        for m = k:dim_hidden(j)
            node_num_k = sum(dim_hidden(1:j-1)) + k;
            node_num_m = sum(dim_hidden(1:j-1)) + m;

            if any(node_num_k == Ip) && any(node_num_m == Ip)
                eq_constraints{ecount} = x_curr_layer(k) - x_curr_layer(m) - (v{j}(k) - v{j}(m)); ecount = ecount + 1;                
            elseif any(node_num_k == In) && any(node_num_m == In)
                eq_constraints{ecount} = x_curr_layer(k) - x_curr_layer(m); ecount = ecount + 1;
            else
                ineq_constraints{icount} = -(x_curr_layer(k) - x_curr_layer(m) - 0)*(x_curr_layer(k) - x_curr_layer(m) - 1*(v{j}(k) - v{j}(m))); icount = icount + 1;                
            end     
        end
    end
end    
end

% Only for relu, enforcing equality constraints, have realised it won't do
% anything
if repeated == 5 && strcmp(AF, 'relu')
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

            if any(j == Ip) && any(k == Ip)
                eq_constraints{ecount} = x1 - x2 - (v1 - v2); ecount = ecount + 1;                
            elseif any(j == In) && any(k == In)
                eq_constraints{ecount} = x1 - x2; ecount = ecount + 1;
            else
                ineq_constraints{icount} = -(x1 - x2 - 0)*(x1 - x2 - 1*(v1 - v2)); icount = icount + 1;                
            end     
        end
    end
   
end

%% Functions for derivatives
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