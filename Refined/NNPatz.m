function [SOLy,prog] = NNPatz(net,u_min,u_max,repeated,c,con_type,sos_order,sos_type)

% Extract dimensions
dims = net.dims;
dim_in = dims(1); 
dim_hidden = dims(2:end-1);
dim_out = dims(end);

% Extract NN weights and biases
W = net.weights;
b = net.biases;

% Extract Activation Function
AF = net.activation;

% Create symbolic variables
syms u [dim_in,1]
syms x [sum(dim_hidden),1]
%syms y [dim_out,1]
syms y [1,1]

% Set variables and decision varaibles in SOS
vars = [u; x]; 
prog = sosprogram(vars);
prog = sosdecvar(prog,y);

% Input constraints
con_in1 = u - u_min;
con_in2 = u_max - u;

% Function for hidden layer constraints
[eq_constraints, ineq_constraints] = hiddenLayerConstraintsTwoSectors(net,AF,u_min,u_max,u,x,y,dim_in,dim_hidden,dim_out,repeated);

% Output layer constraints
v_out = W{end}*x(end - dim_hidden(end) + 1 : end) + b{end};
if dim_out == 1
    f = -c*(y - v_out);
elseif dim_out == 2
    f = c.'*v_out - y;
end

[prog,expr] = assembleConstraints(prog,vars,ineq_constraints,eq_constraints,con_in1,con_in2,f,net,con_type,sos_order,sos_type);
    
% P-satz refutation
prog = sosineq(prog,expr);
if dim_out == 1
    prog = sossetobj(prog,c*y);
elseif dim_out == 2
    prog = sossetobj(prog,y);
end
solver_opt.solver = 'sdpt3'; % put this outside function eventually
prog = sossolve(prog,solver_opt);
SOLy = sosgetsol(prog,y);

end

%%JUNK
% 
% s = cell(length(ineq_constraints),1);
% for j = 1:length(ineq_constraints)
%     [prog,s{j}] = sospolyvar(prog,monomials(vars,0));
%     prog = sosineq(prog,s{j});
% end
% 
% t = cell(length(eq_constraints),1);
% for j = 1:length(eq_constraints)
%     [prog,t{j}] = sospolyvar(prog,monomials(vars,0));
% end
% 
% s2 = cell(length(ineq_constraints),length(ineq_constraints));
% for j = 1:length(ineq_constraints)
%     for k = 1:length(ineq_constraints)
%         [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0));
%         prog = sosineq(prog,s2{j,k});
%     end
% end
% 
% % Create statement 1 + cone + ideal
% expr = - f;
% 
% % Input layer constraints
% for j = 1:dim_in
%     %[prog,s_in1{j}] = sospolyvar(prog,monomials(vars,0));
%     %prog = sosineq(prog,s_in1{j});
%     %[prog,s_in2{j}] = sospolyvar(prog,monomials(vars,0));
%     %prog = sosineq(prog,s_in2{j});
%     %expr = expr - s_in1{j}*con_in1(j) - s_in2{j}*con_in2(j);
%     
%     [prog,s_in3{j}] = sospolyvar(prog,monomials(vars,0));
%     prog = sosineq(prog,s_in3{j});
%     expr = expr - s_in3{j}*con_in1(j)*con_in2(j);
%    
%     %expr = expr - s_in1{j}*con_in1(j) - s_in2{j}*con_in2(j) - s_in3{j}*con_in1(j)*con_in2(j);
% end
% 
% % Hidden layer constraints
% for j = 1:length(ineq_constraints)
%     expr = expr - s{j}*ineq_constraints{j};
% end
% 
% for j = 1:length(eq_constraints)
%     expr = expr - t{j}*eq_constraints{j};
% end
% 
% for j = 1:length(ineq_constraints)
%     for k = j+1:length(ineq_constraints)
%         expr = expr - s2{j,k}*ineq_constraints{j}*ineq_constraints{k};
%     end
% end
% 
% for j = 1:length(ineq_constraints)
%     for k = 1:dim_in
%         [prog,s3{j,k}] = sospolyvar(prog,monomials(vars,0));
%         prog = sosineq(prog,s3{j,k});
%         expr = expr - s3{j,k}*ineq_constraints{j}*con_in1(k)*con_in2(k);
%     end
% end