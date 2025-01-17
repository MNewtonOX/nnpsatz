function [SOLy,prog] = NNPsatz(net,u_min,u_max,repeated,c,con_type,sos_order,sos_type,solver)

% Extract dimensions
dims = net.dims;
dim_in = dims(1); 
dim_hidden = dims(2:end-1);
dim_out = dims(end);

% Create polynomial variables
mpvar( 'u', [dim_in,1])
mpvar( 'x', [sum(dim_hidden),1])
mpvar( 'y', [1,1])

% Can also use symbolic toolbox
%syms u [dim_in,1]
%syms x [sum(dim_hidden),1]
%syms y [dim_out,1]
%syms y [1,1]

% Set variables and decision varaibles in SOS
vars = [u; x]; 
prog = sosprogram(vars);
prog = sosdecvar(prog,y);

% Input constraints
con_in1 = u - u_min;
con_in2 = u_max - u;

% Function for hidden layer constraints
[eq_constraints, ineq_constraints] = hiddenLayerConstraintsTwoSectors(net,u_min,u_max,u,x);
%[eq_constraints, ineq_constraints] = hiddenLayerConstraintsOneSector(net,u_min,u_max,u,x)

if repeated >= 1
    [eq_rep_constraints,ineq_rep_constraints] = hiddenLayerConstraintsRepeated(net,u_min,u_max,repeated,u,x);
else
    eq_rep_constraints = []; ineq_rep_constraints = [];
end

% Output layer constraints
v_out = net.weights{end}*x(end - dim_hidden(end) + 1 : end) + net.biases{end};
if dim_out == 1
    f = -c*(y - v_out);
elseif dim_out == 2
    f = c.'*v_out - y;
end

% Puts all of the constraints into the SOS program, this depends on the sos order and sos type
[prog,expr] = assembleConstraints(prog,vars,ineq_constraints,eq_constraints,eq_rep_constraints,ineq_rep_constraints,repeated,con_in1,con_in2,f,net,con_type,sos_order,sos_type);
    
% P-satz refutation, minimize the objective
prog = sosineq(prog,expr);
if dim_out == 1
    prog = sossetobj(prog,c*y);
elseif dim_out == 2
    prog = sossetobj(prog,y);
end

% Set solver
solver_opt.solver = solver;

% Solve SOS program
prog = sossolve(prog,solver_opt);
SOLy = sosgetsol(prog,y);

end
