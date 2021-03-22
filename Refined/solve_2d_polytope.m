function [X,Y] = solve_2d_polytope(bound,dim_poly,C)

for i = 1:dim_poly
    B(i,:) = bound(i);        
end

B = double(B);
X = [];
Y = [];

for i=1:dim_poly-1
    tmp = linsolve(C([i,i+1],:),B([i,i+1],1));
    X = [X;tmp(1)];
    Y = [Y;tmp(2)];
end

tmp = linsolve(C([1,dim_poly],:),B([1,dim_poly],1));
X = [X;tmp(1)];
Y = [Y;tmp(2)];

end