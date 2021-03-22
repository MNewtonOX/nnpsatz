% Function for hidden layer constraints
function [eq_constraints, ineq_constraints,eq_rep_constraints,ineq_rep_constraints] = hiddenLayerConstraintsTwoSectorsAlt(net,AF,u_min,u_max,u,x,y,dim_in,dim_hidden,dim_out,repeated,resultsValues)

W = net.weights;
b = net.biases;

% Create cell for constraints
ineq_constraints = {};
eq_constraints = {};
ineq_rep_constraints = {};
eq_rep_constraints = {};

icount = 1; %inequality
ecount = 1; %equality
ircount = 1; %inequality repeated
ercount = 1; %equality repeated

%% ReLU
if strcmp(AF, 'relu')

alpha = 0;
beta = 1;

% Pre-processing step to find active and inactive neurons
%[Y_min2,Y_max2,X_min2,X_max2,~,~] = net.interval_arithmetic(u_min,u_max);
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
        %ineq_constraints{icount} = x_curr_layer(k) - X_min_curr_layer(k); icount = icount + 1;
        %ineq_constraints{icount} = -x_curr_layer(k) + X_max_curr_layer(k); icount = icount + 1;
        ineq_constraints{icount,1} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1;
        
    end
end

end

%% Sigmoid
if strcmp(AF, 'sigmoid')
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
        %rounded = 4*[Y_min,Y_max];
        if 2 < 1%Y_min_curr_layer(k) <= -2.5
        [~,index1] = min(abs(4*resultsValues.bounds(:,1) - 4*Y_min_curr_layer(k)));
        [~,index2] = min(abs(4*resultsValues.bounds(:,2) - 4*Y_max_curr_layer(k)));
        x_ml = resultsValues.minCoord(index1,1);
        x_mu = resultsValues.minCoord(index2,2);
        %index = find(resultsValues.bounds == )
        % Compute upper sector lines, which are tangent to the sigmoid curve
        %x_m = 1.5;
        % Right side upper line L_ub
        syms d1
        d1 = solve(sig(d1)*(1 - sig(d1)) == (sig(x_mu) - sig(d1))/(x_mu - d1));
        grad_L_ub = sig(d1)*(1 - sig(d1));
        c_L_ub = sig(d1) - grad_L_ub*d1;

        % Left side upper line L_lb
        syms d2
        d2 = solve(sig(d2)*(1 - sig(d2)) == (sig(x_ml) - sig(d2))/(x_ml - d2));
        grad_L_lb = sig(d2)*(1 - sig(d2));
        c_L_lb = sig(d2) - grad_L_lb*d2;
        else
            x_mu = 1.5;
            x_ml = -1.5;
            % Right side upper line L_ub
            syms d1
            d1 = solve(sig(d1)*(1 - sig(d1)) == (sig(x_mu) - sig(d1))/(x_mu - d1));
            grad_L_ub = sig(d1)*(1 - sig(d1));
            c_L_ub = sig(d1) - grad_L_ub*d1;

            % Left side upper line L_lb
            syms d2
            d2 = solve(sig(d2)*(1 - sig(d2)) == (sig(x_ml) - sig(d2))/(x_ml - d2));
            grad_L_lb = sig(d2)*(1 - sig(d2));
            c_L_lb = sig(d2) - grad_L_lb*d2;
        end
        
        % Two sector constraints
        if Y_max_curr_layer(k) > 0 && Y_min_curr_layer(k) < 0
            % Sector in right hand plane
            if Y_max_curr_layer(k) > x_mu
                grad1a = (X_max_curr_layer(k) - sig(x_mu))/(Y_max_curr_layer(k) - x_mu);
                c1a = X_max_curr_layer(k) - grad1a*Y_max_curr_layer(k);  
            else
                grad1a = 0; 
                c1a = X_max_curr_layer(k);
            end
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad1a*v{j}(k) + c1a))*((grad_L_ub*v{j}(k) + c_L_ub) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 

            % Sector in left hand plane
            if Y_min_curr_layer(k) < x_ml
                grad2a = (X_min_curr_layer(k) - sig(x_ml))/(Y_min_curr_layer(k) - x_ml);
                c2a = X_min_curr_layer(k) - grad2a*Y_min_curr_layer(k);
            else
                grad2a = 0; 
                c2a = X_min_curr_layer(k);
            end
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad2a*v{j}(k) + c2a))*((grad_L_lb*v{j}(k) + c_L_lb) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
        
        elseif Y_max_curr_layer(k) < 0 && Y_min_curr_layer(k) < 0
            Ysec = (Y_max_curr_layer(k) + Y_min_curr_layer(k))/2;
            Xsec = sig(Ysec);
            
            grad1a = (X_max_curr_layer(k) - Xsec)/(Y_max_curr_layer(k) - Ysec);
            c1a = X_max_curr_layer(k) - grad1a*Y_max_curr_layer(k); 
            
            grad2a = (X_min_curr_layer(k) - Xsec)/(Y_min_curr_layer(k) - Ysec);
            c2a = X_min_curr_layer(k) - grad2a*Y_min_curr_layer(k); 
            
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad2a*v{j}(k) + c2a))*((grad1a*v{j}(k) + c1a) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
            
        elseif Y_max_curr_layer(k) > 0 && Y_min_curr_layer(k) > 0
            Ysec = (Y_max_curr_layer(k) + Y_min_curr_layer(k))/2;
            Xsec = sig(Ysec);
            
            grad1a = (X_max_curr_layer(k) - Xsec)/(Y_max_curr_layer(k) - Ysec);
            c1a = X_max_curr_layer(k) - grad1a*Y_max_curr_layer(k); 
            
            grad2a = (X_min_curr_layer(k) - Xsec)/(Y_min_curr_layer(k) - Ysec);
            c2a = X_min_curr_layer(k) - grad2a*Y_min_curr_layer(k); 
            
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad1a*v{j}(k) + c1a))*((grad2a*v{j}(k) + c2a) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
            
        end
        
%         %Plot the sectors  
%         fplot(1/(1+exp(-x(1))))
%         hold on
%         fplot(grad1a*x(1) + c1a)
%         fplot(grad_L_ub*x(1) + c_L_ub)
%         fplot(grad2a*x(1) + c2a)
%         fplot(grad_L_lb*x(1) + c_L_lb)

        % 0, 0.25 sector for test
        %ineq_constraints{icount} = (x_curr_layer(k) - (0*v{j}(k) + 0.5))*((0.25*v{j}(k) + 0.5) - x_curr_layer(k)); icount = icount + 1;
        
        % Pre-processing bounds
        ineq_constraints{icount,1} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
    end
end
end

%% tanh
if strcmp(AF, 'tanh')
%[Y_min,Y_max,X_min,X_max,~,~] = net.interval_arithmetic(u_min,u_max);
[Y_min,Y_max,X_min,X_max] = interval_bound_propigation(u_min,u_max,dim_hidden,net);

% Compute upper sector lines, which are tangent to the sigmoid curve
x_m = 0.75;%1.5;
% Right side upper line L_ub
syms d1
d1 = vpasolve((1 - (tanh(d1))^2) == (tanh(x_m) - tanh(d1))/(x_m - d1),d1,[-10,0]);
grad_L_ub = 1 - (tanh(d1))^2;
c_L_ub = tanh(d1) - grad_L_ub*d1;

% Left side upper line L_lb
syms d2
d2 = vpasolve((1 - (tanh(d2))^2) == (tanh(-x_m) - tanh(d2))/(-x_m - d2),d2,[0,10]);
grad_L_lb = 1 - (tanh(d2))^2;
c_L_lb = tanh(d2) - grad_L_lb*d2;

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

        % Two sector constraints
        if Y_max_curr_layer(k) > 0 && Y_min_curr_layer(k) < 0
            % Sector in right hand plane 
            if Y_max_curr_layer(k) > x_m
                grad1a = (X_max_curr_layer(k) - tanh(x_m))/(Y_max_curr_layer(k) - x_m);
                c1a = X_max_curr_layer(k) - grad1a*Y_max_curr_layer(k);  
            else
                grad1a = 0; 
                c1a = X_max_curr_layer(k);
            end
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad1a*v{j}(k) + c1a))*((grad_L_ub*v{j}(k) + c_L_ub) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 

            % Sector in left hand plane 
            if Y_min_curr_layer(k) < -x_m
                grad2a = (X_min_curr_layer(k) - tanh(-x_m))/(Y_min_curr_layer(k) - -x_m);
                c2a = X_min_curr_layer(k) - grad2a*Y_min_curr_layer(k);
            else
                grad2a = 0; 
                c2a = X_min_curr_layer(k);
            end
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad2a*v{j}(k) + c2a))*((grad_L_lb*v{j}(k) + c_L_lb) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
        
        elseif Y_max_curr_layer(k) < 0 && Y_min_curr_layer(k) < 0
            Ysec = (Y_max_curr_layer(k) + Y_min_curr_layer(k))/2;
            Xsec = tanh(Ysec);
            
            grad1a = (X_max_curr_layer(k) - Xsec)/(Y_max_curr_layer(k) - Ysec);
            c1a = X_max_curr_layer(k) - grad1a*Y_max_curr_layer(k); 
            
            grad2a = (X_min_curr_layer(k) - Xsec)/(Y_min_curr_layer(k) - Ysec);
            c2a = X_min_curr_layer(k) - grad2a*Y_min_curr_layer(k); 
            
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad2a*v{j}(k) + c2a))*((grad1a*v{j}(k) + c1a) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
            
        elseif Y_max_curr_layer(k) > 0 && Y_min_curr_layer(k) > 0
            Ysec = (Y_max_curr_layer(k) + Y_min_curr_layer(k))/2;
            Xsec = tanh(Ysec);
            
            grad1a = (X_max_curr_layer(k) - Xsec)/(Y_max_curr_layer(k) - Ysec);
            c1a = X_max_curr_layer(k) - grad1a*Y_max_curr_layer(k); 
            
            grad2a = (X_min_curr_layer(k) - Xsec)/(Y_min_curr_layer(k) - Ysec);
            c2a = X_min_curr_layer(k) - grad2a*Y_min_curr_layer(k); 
            
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad1a*v{j}(k) + c1a))*((grad2a*v{j}(k) + c2a) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
            
        end

        % Plot sectors to test constraint
        fplot(tanh(x(1)))
        hold on
        fplot(grad1a*x(1) + c1a)
        fplot(grad_L_ub*x(1) + c_L_ub)
        fplot(grad2a*x(1) + c2a)
        fplot(grad_L_lb*x(1) + c_L_lb)
               
        % Pre-processing bounds
        ineq_constraints{icount,1} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
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
            ineq_rep_constraints{ircount} = ((x2 - x1) - 0*(v2 - v1))*(1*(v2 - v1) - (x2 - x1)); ircount = ircount + 1;
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
            ineq_rep_constraints{ircount} = -(x_curr_layer(k) - x_curr_layer(m) - 0)*(x_curr_layer(k) - x_curr_layer(m) - 1*(v{j}(k) - v{j}(m))); ircount = ircount + 1;  
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
            ineq_rep_constraints{ircount} = -(x_curr_layer(k) - x_curr_layer(m) - 0)*(x_curr_layer(k) - x_curr_layer(m) - 1*(v{j}(k) - v{j}(m))); ircount = ircount + 1;  
        end
    end
    if j > 1
        for k = 1:dim_hidden(j)
            for m = 1:dim_hidden(j-1)
                ineq_rep_constraints{ircount} = -(x_curr_layer(k) - x_prev_layer(m) - 0)*(x_curr_layer(k) - x_prev_layer(m) - 1*(v{j}(k) - v{j-1}(m))); ircount = ircount + 1;
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
                eq_rep_constraints{ercount} = x_curr_layer(k) - x_curr_layer(m) - (v{j}(k) - v{j}(m)); ercount = ercount + 1;                
            elseif any(node_num_k == In) && any(node_num_m == In)
                eq_rep_constraints{ercount} = x_curr_layer(k) - x_curr_layer(m); ercount = ercount + 1;
            else
                ineq_rep_constraints{ircount} = -(x_curr_layer(k) - x_curr_layer(m) - 0)*(x_curr_layer(k) - x_curr_layer(m) - 1*(v{j}(k) - v{j}(m))); ircount = ircount + 1;                
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
                eq_rep_constraints{ercount} = x1 - x2 - (v1 - v2); ercount = ercount + 1;                
            elseif any(j == In) && any(k == In)
                eq_rep_constraints{ercount} = x1 - x2; ercount = ercount + 1;
            else
                ineq_rep_constraints{ircount} = -(x1 - x2 - 0)*(x1 - x2 - 1*(v1 - v2)); ircount = ircount + 1;                
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