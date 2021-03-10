function [prog,expr] = assembleConstraints(prog,vars,ineq_constraints,eq_constraints,con_in1,con_in2,f,net,con_type,sos_order,sos_type)

% Extract dimensions
dims = net.dims;
dim_in = dims(1); 
dim_hidden = dims(2:end-1);
dim_out = dims(end);

% Set order to something easy
a = sos_order;

% Create statement 1 + cone + ideal
expr = - f;

% Add all constraints and their multipliers together
s = cell(size(ineq_constraints,1),1);
for j = 1:size(ineq_constraints,1)
    [prog,s{j}] = setPolynomials(prog,vars,a,sos_type,ineq_constraints{j,1},ineq_constraints{j,2},dims);
    %[prog,s{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s{j});
    expr = expr - s{j}*ineq_constraints{j,1};
end

t = cell(size(eq_constraints,1),1);
for j = 1:size(eq_constraints,1)
    [prog,t{j}] = setPolynomials(prog,vars,a,sos_type,eq_constraints{j,1},eq_constraints{j,2},dims);
    %[prog,t{j}] = sospolyvar(prog,monomials(vars,0:a));
    expr = expr - t{j}*eq_constraints{j,1};
end

% Input layer constraints
s_in = cell(dim_in,1);
for j = 1:dim_in
    [prog,s_in{j}] = setPolynomials(prog,vars,a,sos_type,con_in1(j)*con_in2(j),[0,j],dims);
    prog = sosineq(prog,s_in{j});
    expr = expr - s_in{j}*con_in1(j)*con_in2(j);
end

%end

if con_type == 1 || con_type == 2

% Multiply all constraints by one other constraint, excluding input constraints
s2 = cell(size(ineq_constraints,1),size(ineq_constraints,1));
for j = 1:size(ineq_constraints,1)
    for k = j+1:size(ineq_constraints,1)
        [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0:a));
        prog = sosineq(prog,s2{j,k});
        expr = expr - s2{j,k}*ineq_constraints{j,1}*ineq_constraints{k,1};
    end
end

end

% Multiply all constraints by one other constraint
if con_type == 2

% Multiple all constraits by input constraints
s3 = cell(size(ineq_constraints,1),dim_in);
s4 = cell(size(ineq_constraints,1),dim_in);
for j = 1:size(ineq_constraints,1)
    for k = 1:dim_in
        [prog,s3{j,k}] = sospolyvar(prog,monomials(vars,0:a));
        prog = sosineq(prog,s3{j,k});
        [prog,s4{j,k}] = sospolyvar(prog,monomials(vars,0:a));
        prog = sosineq(prog,s4{j,k});
        %expr = expr - s3{j,k}*ineq_constraints{j}*con_in1(k)*con_in2(k);
        expr = expr - s3{j,k}*ineq_constraints{j,1}*con_in1(k);
        expr = expr - s4{j,k}*ineq_constraints{j,1}*con_in2(k);
    end
end

end

% Multiply all constraints within the same node
if con_type == 3

% Multiple constraints together
%s2 = cell(size(ineq_constraints,1),size(ineq_constraints,1));
for j = 1:size(ineq_constraints,1)
    for k = j+1:size(ineq_constraints,1)
        if ineq_constraints{j,2}(1) == ineq_constraints{k,2}(1)
            if ineq_constraints{j,2}(2) == ineq_constraints{k,2}(2)
                [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0:a));
                prog = sosineq(prog,s2{j,k});
                expr = expr - s2{j,k}*ineq_constraints{j,1}*ineq_constraints{k,1};       
            end
        end
    end
end

end

% Multiply all constraints within the same layer
if con_type == 4

% Multiple constraints together
%s2 = cell(size(ineq_constraints,1),size(ineq_constraints,1));
for j = 1:size(ineq_constraints,1)
    for k = j+1:size(ineq_constraints,1)
        if ineq_constraints{j,2}(1) == ineq_constraints{k,2}(1)          
            [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0:a));
            prog = sosineq(prog,s2{j,k});
            expr = expr - s2{j,k}*ineq_constraints{j,1}*ineq_constraints{k,1};       
        end
    end
end

end


if con_type == 5 || con_type == 6

% Multiply all constraints within the same layer and neighbouring layers
for j = 1:size(ineq_constraints,1)
    for k = j+1:size(ineq_constraints,1)
        if (ineq_constraints{j,2}(1) == ineq_constraints{k,2}(1)) || (ineq_constraints{j,2}(1) == ineq_constraints{k,2}(1) - 1)          
            [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0:a));
            prog = sosineq(prog,s2{j,k});
            expr = expr - s2{j,k}*ineq_constraints{j,1}*ineq_constraints{k,1};       
        end
    end
end

end
 
if con_type == 5
% Multiply first layer by input layer
for j = 1:size(ineq_constraints,1) 
    if ineq_constraints{j,2}(1) == 1
        for k = 1:dim_in
            [prog,s3{j}] = sospolyvar(prog,monomials(vars,0:a));
            prog = sosineq(prog,s3{j});
            [prog,s4{j}] = sospolyvar(prog,monomials(vars,0:a));
            prog = sosineq(prog,s4{j});
            expr = expr - s3{j}*ineq_constraints{j,1}*con_in1(k);
            expr = expr - s4{j}*ineq_constraints{j,1}*con_in2(k);    
        end
    end
end

end

end

function [prog,s] = setPolynomials(prog,vars,a,sos_type,constraints,node_number,dims)

dim_in = dims(1); 
dim_hidden = dims(2:end-1);
dim_out = dims(end);

if sos_type == 0
   [prog,s] = sospolyvar(prog,monomials(vars,0:a));
elseif sos_type == 1
    vars = symvar(constraints);
    [prog,s] = sospolyvar(prog,monomials(vars,0:a));
elseif sos_type == 2    
    var_number = dim_in + sum(dim_hidden(1:node_number(1)-1)) + node_number(2);
    vars = vars(var_number);
    [prog,s] = sospolyvar(prog,monomials(vars,0:a));
elseif sos_type == 3
    if node_number(1) > 0
        vars_layer = (dim_in + 1 + sum(dim_hidden(1:(node_number(1) - 1)))):(dim_in + sum(dim_hidden(1:(node_number(1) + 0))) - 0);
    else
        vars_layer = 1:dim_in;
    end
    vars = vars(vars_layer);
    [prog,s] = sospolyvar(prog,monomials(vars,0:a));
elseif sos_type == 4
    if node_number(1) > 1
        if node_number(1) + 1 <= length(dim_hidden)
            vars_layer = (dim_in + 1 + sum(dim_hidden(1:(node_number(1) - 2)))):(dim_in + sum(dim_hidden(1:(node_number(1) + 1))) - 0);
        else
            vars_layer = (dim_in + 1 + sum(dim_hidden(1:(node_number(1) - 2)))):(dim_in + sum(dim_hidden(1:end)) - 0);
        end
    elseif node_number(1) == 1
        vars_layer = (dim_in + 1 + sum(dim_hidden(1:(node_number(1) - 2)))):(dim_in + sum(dim_hidden(1:(node_number(1) + 1))) - 0);
        vars_layer = [1, vars_layer];
    elseif node_number(1) == 0
        vars_layer = 1:(dim_in + sum(dim_hidden(1)));
    end
    vars = vars(vars_layer);
    [prog,s] = sospolyvar(prog,monomials(vars,0:a));
end
    
end






%%Old


% % Hidden layer constraints
% for j = 1:length(ineq_constraints)
%     
% end
% 
% for j = 1:length(eq_constraints)
%     
% end


% s2 = cell(length(ineq_constraints),length(ineq_constraints));
% for j = 1:length(ineq_constraints)
%     for k = 1:length(ineq_constraints)
%         [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0));
%         prog = sosineq(prog,s2{j,k});
%     end
% end

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

% for j = 1:length(ineq_constraints)
%     for k = j+1:length(ineq_constraints)
%         expr = expr - s2{j,k}*ineq_constraints{j}*ineq_constraints{k};
%     end
% end

% for j = 1:length(ineq_constraints)
%     for k = 1:dim_in
%         [prog,s3{j,k}] = sospolyvar(prog,monomials(vars,0));
%         prog = sosineq(prog,s3{j,k});
%         expr = expr - s3{j,k}*ineq_constraints{j}*con_in1(k)*con_in2(k);
%     end
% end