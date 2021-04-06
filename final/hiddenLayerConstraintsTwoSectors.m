% Function for hidden layer constraints
function [eq_constraints, ineq_constraints] = hiddenLayerConstraintsTwoSectors(net,u_min,u_max,u,x)

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

% Counters
icount = 1; %inequality
ecount = 1; %equality

%% ReLU
if strcmp(AF, 'relu')

% Sector constraint values
alpha = 0;
beta = 1;

% Pre-processing step to find active and inactive neurons
[Y_min,Y_max,X_min,X_max] = intervalBoundPropagation(u_min,u_max,dim_hidden,net);
Ip = find(Y_min>1e-3);
In = find(Y_max<-1e-3);
Ipn = setdiff(1:sum(dim_hidden),union(Ip,In));

v = cell(length(dim_hidden),1);
eq_constraints = cell(2*length(Ip) + 2*length(In) + length(Ipn),2);
ineq_constraints = cell(length(Ip) + length(In) + 2*length(Ipn),2);
for j = 1:length(dim_hidden)
    if j == 1
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
        
        % Pre-processing bounds
        ineq_constraints{icount,1} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1;
        
    end
end

end

%% Sigmoid
if strcmp(AF, 'sigmoid')
    
% Pre-procesing step to obtain approximate bounds    
[Y_min,Y_max,X_min,X_max] = intervalBoundPropagation(u_min,u_max,dim_hidden,net);

% Compute upper sector lines, which are tangent to the sigmoid curve

% Set mid point of the sector, which is a hyper-parameter that can be tuned
x_m = 1; 

% Right side upper line L_ub
syms d1
d1 = solve(sig(d1)*(1 - sig(d1)) == (sig(x_m) - sig(d1))/(x_m - d1));
grad_L_ub = sig(d1)*(1 - sig(d1));
grad_L_ub = double(grad_L_ub);
c_L_ub = sig(d1) - grad_L_ub*d1;
c_L_ub = double(c_L_ub);

% Left side upper line L_lb
syms d2
d2 = solve(sig(d2)*(1 - sig(d2)) == (sig(-x_m) - sig(d2))/(-x_m - d2));
grad_L_lb = sig(d2)*(1 - sig(d2));
grad_L_lb = double(grad_L_lb);
c_L_lb = sig(d2) - grad_L_lb*d2;
c_L_lb = double(c_L_lb);

%ineq_constraints = cell(3*sum(dim_hidden),2);
v = cell(length(dim_hidden),1);
for j = 1:length(dim_hidden)
    if j == 1
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
                grad1a = (X_max_curr_layer(k) - sig(x_m))/(Y_max_curr_layer(k) - x_m);
                c1a = X_max_curr_layer(k) - grad1a*Y_max_curr_layer(k);  
                % Check for overlapping sectors
                if X_min_curr_layer(k) >  Y_min_curr_layer(k)*grad1a + c1a
                    grad1a = (X_min_curr_layer(k) - sig(x_m))/(Y_min_curr_layer(k) - x_m);
                    c1a = X_min_curr_layer(k) - grad1a*Y_min_curr_layer(k);  
                end
            else
                grad1a = 0; 
                %c1a = X_max_curr_layer(k);
                c1a = sig(x_m);
            end
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad1a*v{j}(k) + c1a))*((grad_L_ub*v{j}(k) + c_L_ub) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 

            % Sector in left hand plane
            if Y_min_curr_layer(k) < -x_m
                grad2a = (X_min_curr_layer(k) - sig(-x_m))/(Y_min_curr_layer(k) - -x_m);
                c2a = X_min_curr_layer(k) - grad2a*Y_min_curr_layer(k);
                % Check for overlapping sectors
                if X_max_curr_layer(k) <  Y_max_curr_layer(k)*grad2a + c2a
                    grad2a = (X_max_curr_layer(k) - sig(-x_m))/(Y_max_curr_layer(k) - -x_m);
                    c2a = X_max_curr_layer(k) - grad2a*Y_max_curr_layer(k);  
                end
            else
                grad2a = 0; 
                %c2a = X_min_curr_layer(k);
                c2a = sig(-x_m);
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
        
        % Can be uncommented to plot the sectors and inspect the constraints
%         fplot(1/(1+exp(-x(1))))
%         hold on
%         fplot(grad1a*x(1) + c1a)
%         fplot(grad_L_ub*x(1) + c_L_ub)
%         fplot(grad2a*x(1) + c2a)
%         fplot(grad_L_lb*x(1) + c_L_lb)
        
        % Pre-processing bounds
        ineq_constraints{icount,1} = (x_curr_layer(k) - X_min_curr_layer(k))*(-x_curr_layer(k) + X_max_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 
    end
end
end

%% tanh
if strcmp(AF, 'tanh')

% Pre-procesing step to obtain approximate bounds    
[Y_min,Y_max,X_min,X_max] = intervalBoundPropagation(u_min,u_max,dim_hidden,net);

% Compute upper sector lines, which are tangent to the sigmoid curve

% Set mid point of the sector, which is a hyper-parameter that can be tuned
x_m = 1.1;

% Right side upper line L_ub
syms d1
d3 = d1;
d1 = vpasolve((1 - (tanh(d1))^2) == (tanh(x_m) - tanh(d1))/(x_m - d1),d1,[-10,0]);
grad_L_ub = 1 - (tanh(d1))^2;
grad_L_ub = double(grad_L_ub);
c_L_ub = tanh(d1) - grad_L_ub*d1;
c_L_ub = double(c_L_ub);

% Left side upper line L_lb
syms d2
d2 = vpasolve((1 - (tanh(d2))^2) == (tanh(-x_m) - tanh(d2))/(-x_m - d2),d2,[0,10]);
grad_L_lb = 1 - (tanh(d2))^2;
grad_L_lb = double(grad_L_lb);
c_L_lb = tanh(d2) - grad_L_lb*d2;
c_L_lb = double(c_L_lb);

for j = 1:length(dim_hidden)
    if j == 1
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
                % Check for overlapping sectors
                if X_min_curr_layer(k) >  Y_min_curr_layer(k)*grad1a + c1a
                    grad1a = (X_min_curr_layer(k) - tanh(x_m))/(Y_min_curr_layer(k) - x_m);
                    c1a = X_min_curr_layer(k) - grad1a*Y_min_curr_layer(k);  
                end
            else
                grad1a = 0; 
                %c1a = X_max_curr_layer(k);
                c1a = tanh(x_m);
            end
            ineq_constraints{icount,1} = (x_curr_layer(k) - (grad1a*v{j}(k) + c1a))*((grad_L_ub*v{j}(k) + c_L_ub) - x_curr_layer(k)); ineq_constraints{icount,2} = [j,k]; icount = icount + 1; 

            % Sector in left hand plane 
            if Y_min_curr_layer(k) < -x_m
                grad2a = (X_min_curr_layer(k) - tanh(-x_m))/(Y_min_curr_layer(k) - -x_m);
                c2a = X_min_curr_layer(k) - grad2a*Y_min_curr_layer(k);
                % Check for overlapping sectors
                if X_max_curr_layer(k) <  Y_max_curr_layer(k)*grad2a + c2a
                    grad2a = (X_max_curr_layer(k) - tanh(-x_m))/(Y_max_curr_layer(k) - -x_m);
                    c2a = X_max_curr_layer(k) - grad2a*Y_max_curr_layer(k);  
                end
            else
                grad2a = 0; 
                %c2a = X_min_curr_layer(k);
                c2a = tanh(-x_m);
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

        % Can be uncommented to plot the sectors and inspect the constraints
%         fplot(tanh(d3))
%         hold on
%         fplot(grad1a*d3 + c1a)
%         fplot(grad_L_ub*d3 + c_L_ub)
%         fplot(grad2a*d3 + c2a)
%         fplot(grad_L_lb*d3 + c_L_lb)
               
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