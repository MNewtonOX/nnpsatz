%syms a x_m
clear all
clc

%sigm = 1/(1+exp(-a));

x_u = 4;
x_l = -0.5;

k = 1;
for x_m = x_l:0.1:x_u

area1(k) = (sig(x_u) - sig(x_m))*(x_u - x_m) - (sig(x_u) - sig(x_m))^2*4;

p2 = 4*(sig(x_u) - sig(x_m)) + x_m;
p3 = (sig(x_l) - sig(x_u))*(x_u - x_m)/(sig(x_u) - sig(x_m)) + x_u;
p4 = (sig(x_u) - sig(x_m))*(x_l - x_u)/(x_u - x_m) + sig(x_u);

area2(k) = (sig(x_m) - sig(x_l))*(p2 - p3) - (x_l - p3)*(p4 - sig(x_l));

total_area(k) = area1(k) + area2(k); k = k + 1;

end

function y = sig(x)
    y = 1/(1+exp(-x));
end


% m1 = (subs(sig,x_u) - subs(sig,x_m))/(x_u - x_m);
% c1 = subs(sig,x_u) - x_u/(x_u - x_m);
% %line1 = m1*a + c1;
% 
% m2 = 0.25;
% c2 = subs(sig,x_m) - x_m*0.25;
% %line2 = m2*a + c2;
% 
% area1 = sig()
% 
% %int(line2 - line1, x_m, x_u)