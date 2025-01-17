% Plot sectors for paper

AF = 'sigmoid';

if strcmp(AF, 'sigmoid')

%[Y_min,Y_max,X_min,X_max] = interval_bound_propigation(u_min,u_max,dim_hidden,net);
Y_min = -5;
Y_max = 5;
X_min = sig(Y_min);
X_max = sig(Y_max);

% Compute upper sector lines, which are tangent to the sigmoid curve
x_m = 1.5;
% Right side upper line L_ub
syms d1
d1 = vpasolve((sig(d1)*(1 - sig(d1)) == (sig(x_m) - sig(d1))/(x_m - d1)),d1,[-10,0]);
grad_L_ub = sig(d1)*(1 - sig(d1));
c_L_ub = sig(d1) - grad_L_ub*d1;

% Left side upper line L_lb
syms d2
d2 = vpasolve(sig(d2)*(1 - sig(d2)) == (sig(-x_m) - sig(d2))/(-x_m - d2));
grad_L_lb = sig(d2)*(1 - sig(d2));
c_L_lb = sig(d2) - grad_L_lb*d2;

% Right side lower line
grad1a = (X_max - sig(x_m))/(Y_max - x_m);
c1a = X_max - grad1a*Y_max;  

% Left side lower line
grad2a = (X_min - sig(-x_m))/(Y_min - -x_m);
c2a = X_min - grad2a*Y_min;

%Plot the sectors  
syms x
fplot(1/(1+exp(-x)),[-6,6],'black','LineWidth',2)
hold on
fplot(grad1a*x + c1a,[-6,6],'red','LineWidth',2)
fplot(grad_L_ub*x + c_L_ub,'red',[-6,6],'LineWidth',2)
fplot(grad2a*x + c2a,[-6,6],'blue','LineWidth',2)
fplot(grad_L_lb*x + c_L_lb,[-6,6],'blue','LineWidth',2)

plot(Y_min,X_min,'g*')
plot(Y_max,X_max,'g*')

plot([Y_max,Y_max], [0,1],'g')
plot([Y_min,Y_min], [0,1],'g')

plot([-6,6], [X_max,X_max],'g')
plot([-6,6], [X_min,X_min],'g')

end


if strcmp(AF, 'tanh')
    
Y_min = -5;
Y_max = 5;
X_min = tanh(Y_min);
X_max = tanh(Y_max);

% Compute upper sector lines, which are tangent to the sigmoid curve
x_m = 1.5;
% Right side upper line L_ub
syms d1
d1 = vpasolve((1 - tanh(d1)^2 == (tanh(x_m) - tanh(d1))/(x_m - d1)),d1,[-10,0]);
grad_L_ub = 1 - tanh(d1)^2;
c_L_ub = tanh(d1) - grad_L_ub*d1;

% Left side upper line L_lb
syms d2
d2 = vpasolve(1 - tanh(d1)^2 == (tanh(-x_m) - tanh(d2))/(-x_m - d2),d2,[-10,0]);
grad_L_lb = 1 - tanh(d1)^2;
c_L_lb = tanh(d2) - grad_L_lb*d2;

% Right side lower line
grad1a = (X_max - tanh(x_m))/(Y_max - x_m);
c1a = X_max - grad1a*Y_max;  

% Left side lower line
grad2a = (X_min - tanh(-x_m))/(Y_min - -x_m);
c2a = X_min - grad2a*Y_min;

%Plot the sectors  
syms x
fplot(tanh(x),[-6,6],'black','LineWidth',2)
hold on
fplot(grad1a*x + c1a,[-6,6],'red','LineWidth',2)
fplot(grad_L_ub*x + c_L_ub,'red',[-6,6],'LineWidth',2)
fplot(grad2a*x + c2a,[-6,6],'blue','LineWidth',2)
fplot(grad_L_lb*x + c_L_lb,[-6,6],'blue','LineWidth',2)   

plot(Y_min,X_min,'g*')
plot(Y_max,X_max,'g*')
    
end


if strcmp(AF, 'relu')

Y_min = -5;
Y_max = 5;

fplot(max(0,x),[-6,6])    
hold on
%fplot(0,[-6,0]) 
%hold on
%fplot(x,[0,6]) 
    
plot(Y_min,X_min,'g*')
plot(Y_max,X_max,'g*')

end

function [y] = difsig(x)
    y = sig(x)*(1 - sig(x));
end

function [y] = sig(x)
    y = 1./(1+exp(-x));
end

function [y] = diftanh(x)
    y = 1 - tanh(x)^2;
end