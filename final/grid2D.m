function [Xin] = grid2D(u_min,u_max)

x1 = linspace(u_min(1),u_max(1),500);
x2 = linspace(u_min(2),u_max(2),500);

[X1,X2] = meshgrid(x1,x2);

Xin(1,:) = X1(:);
Xin(2,:) = X2(:);

end