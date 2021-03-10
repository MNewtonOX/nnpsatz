function [prog,expr] = assembleConstraints(prog,vars,ineq_constraints,eq_constraints,con_in1,con_in2,f,net,con_type,sos_order,sos_type)

% Extract dimensions
dims = net.dims;
dim_in = dims(1); 
dim_hidden = dims(2:end-1);
dim_out = dims(end);

a = sos_order;

% Create statement 1 + cone + ideal
expr = - f;

% Multiply all constraints by a scaler. No overlapping constraints. This is the most computational
% efficient method
if con_type == 0

s = cell(size(ineq_constraints,1),1);
for j = 1:size(ineq_constraints,1)
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s{j});
    expr = expr - s{j}*ineq_constraints{j,1};
end

t = cell(size(eq_constraints,1),1);
for j = 1:size(eq_constraints,1)
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0:a));
    expr = expr - t{j}*eq_constraints{j,1};
end

% Input layer constraints
s_in = cell(dim_in,1);
for j = 1:dim_in
    [prog,s_in{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s_in{j});
    expr = expr - s_in{j}*con_in1(j)*con_in2(j);
end

end

% Multiply all constraints by a scaler. Multiply all constraints by one
% other constraint, excluding input constraints
if con_type == 1

s = cell(size(ineq_constraints,1),1);
for j = 1:size(ineq_constraints,1)
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s{j});
    expr = expr - s{j}*ineq_constraints{j,1};
end

t = cell(size(eq_constraints,1),1);
for j = 1:size(eq_constraints,1)
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0:a));
    expr = expr - t{j}*eq_constraints{j,1};
end

% Input layer constraints
s_in = cell(dim_in,1);
for j = 1:dim_in
    [prog,s_in{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s_in{j});
    expr = expr - s_in{j}*con_in1(j)*con_in2(j);
end

% Multiple constraints together
s2 = cell(size(ineq_constraints,1),size(ineq_constraints,1));
for j = 1:size(ineq_constraints,1)
    for k = j+1:size(ineq_constraints,1)
        [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0:a));
        prog = sosineq(prog,s2{j,k});
        expr = expr - s2{j,k}*ineq_constraints{j,1}*ineq_constraints{k,1};
    end
end

end

% Multiply all constraints by a scaler. Multiply all constraints by one
% other constraint
if con_type == 2

s = cell(size(ineq_constraints,1),1);
for j = 1:size(ineq_constraints,1)
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s{j});
    expr = expr - s{j}*ineq_constraints{j,1};
end

t = cell(size(eq_constraints,1),1);
for j = 1:size(eq_constraints,1)
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0:a));
    expr = expr - t{j}*eq_constraints{j,1};
end

% Input layer constraints
s_in = cell(dim_in,1);
for j = 1:dim_in
    [prog,s_in{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s_in{j});
    expr = expr - s_in{j}*con_in1(j)*con_in2(j);
end

% Multiple constraints together
s2 = cell(size(ineq_constraints,1),size(ineq_constraints,1));
for j = 1:size(ineq_constraints,1)
    for k = j+1:size(ineq_constraints,1)
        [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0:a));
        prog = sosineq(prog,s2{j,k});
        expr = expr - s2{j,k}*ineq_constraints{j,1}*ineq_constraints{k,1};
    end
end

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

% Multiply all constraints by a scaler. Multiply all constraints within the
% same node
if con_type == 3

s = cell(size(ineq_constraints,1),1);
for j = 1:size(ineq_constraints,1)
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s{j});
    expr = expr - s{j}*ineq_constraints{j,1};
end

t = cell(size(eq_constraints,1),1);
for j = 1:size(eq_constraints,1)
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0:a));
    expr = expr - t{j}*eq_constraints{j,1};
end

% Input layer constraints
s_in = cell(dim_in,1);
for j = 1:dim_in
    [prog,s_in{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s_in{j});
    expr = expr - s_in{j}*con_in1(j)*con_in2(j);
end

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

% % Multiple all constraits by input constraints
% s3 = cell(size(ineq_constraints,1),dim_in);
% s4 = cell(size(ineq_constraints,1),dim_in);
% for j = 1:size(ineq_constraints,1)
%     for k = 1:dim_in
%         [prog,s3{j,k}] = sospolyvar(prog,monomials(vars,0:a));
%         prog = sosineq(prog,s3{j,k});
%         [prog,s4{j,k}] = sospolyvar(prog,monomials(vars,0:a));
%         prog = sosineq(prog,s4{j,k});
%         %expr = expr - s3{j,k}*ineq_constraints{j}*con_in1(k)*con_in2(k);
%         expr = expr - s3{j,k}*ineq_constraints{j,1}*con_in1(k);
%         expr = expr - s4{j,k}*ineq_constraints{j,1}*con_in2(k);
%     end
% end

end

% Multiply all constraints by a scaler. Multiply all constraints within the
% same layer
if con_type == 4

s = cell(size(ineq_constraints,1),1);
for j = 1:size(ineq_constraints,1)
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s{j});
    expr = expr - s{j}*ineq_constraints{j,1};
end

t = cell(size(eq_constraints,1),1);
for j = 1:size(eq_constraints,1)
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0:a));
    expr = expr - t{j}*eq_constraints{j,1};
end

% Input layer constraints
s_in = cell(dim_in,1);
for j = 1:dim_in
    [prog,s_in{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s_in{j});
    expr = expr - s_in{j}*con_in1(j)*con_in2(j);
end

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

% % Multiple all constraits by input constraints
% s3 = cell(size(ineq_constraints,1),dim_in);
% s4 = cell(size(ineq_constraints,1),dim_in);
% for j = 1:size(ineq_constraints,1)
%     for k = 1:dim_in
%         [prog,s3{j,k}] = sospolyvar(prog,monomials(vars,0:a));
%         prog = sosineq(prog,s3{j,k});
%         [prog,s4{j,k}] = sospolyvar(prog,monomials(vars,0:a));
%         prog = sosineq(prog,s4{j,k});
%         %expr = expr - s3{j,k}*ineq_constraints{j}*con_in1(k)*con_in2(k);
%         expr = expr - s3{j,k}*ineq_constraints{j,1}*con_in1(k);
%         expr = expr - s4{j,k}*ineq_constraints{j,1}*con_in2(k);
%     end
% end

end

% Multiply all constraints by a scaler. Multiply all constraints within the
% same layer and neighbouring layers
if con_type == 5

s = cell(size(ineq_constraints,1),1);
for j = 1:size(ineq_constraints,1)
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s{j});
    expr = expr - s{j}*ineq_constraints{j,1};
end

t = cell(size(eq_constraints,1),1);
for j = 1:size(eq_constraints,1)
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0:a));
    expr = expr - t{j}*eq_constraints{j,1};
end

% Input layer constraints
s_in = cell(dim_in,1);
for j = 1:dim_in
    [prog,s_in{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s_in{j});
    expr = expr - s_in{j}*con_in1(j)*con_in2(j);
end

% Multiple constraints together
%s2 = cell(size(ineq_constraints,1),size(ineq_constraints,1));
for j = 1:size(ineq_constraints,1)
    for k = j+1:size(ineq_constraints,1)
        if (ineq_constraints{j,2}(1) == ineq_constraints{k,2}(1)) || (ineq_constraints{j,2}(1) == ineq_constraints{k,2}(1) - 1)          
            [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0:a));
            prog = sosineq(prog,s2{j,k});
            expr = expr - s2{j,k}*ineq_constraints{j,1}*ineq_constraints{k,1};       
        end
    end
end
 
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

% Multiply all constraints within the same layer and neighbouring layers,
% excluding input constraints
if con_type == 6

s = cell(size(ineq_constraints,1),1);
for j = 1:size(ineq_constraints,1)
    [prog,s{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s{j});
    expr = expr - s{j}*ineq_constraints{j,1};
end

t = cell(size(eq_constraints,1),1);
for j = 1:size(eq_constraints,1)
    [prog,t{j}] = sospolyvar(prog,monomials(vars,0:a));
    expr = expr - t{j}*eq_constraints{j,1};
end

% Input layer constraints
s_in = cell(dim_in,1);
for j = 1:dim_in
    [prog,s_in{j}] = sospolyvar(prog,monomials(vars,0:a));
    prog = sosineq(prog,s_in{j});
    expr = expr - s_in{j}*con_in1(j)*con_in2(j);
end

% Multiple constraints together
%s2 = cell(size(ineq_constraints,1),size(ineq_constraints,1));
for j = 1:size(ineq_constraints,1)
    for k = j+1:size(ineq_constraints,1)
        if (ineq_constraints{j,2}(1) == ineq_constraints{k,2}(1)) || (ineq_constraints{j,2}(1) == ineq_constraints{k,2}(1) - 1)          
            [prog,s2{j,k}] = sospolyvar(prog,monomials(vars,0:a));
            prog = sosineq(prog,s2{j,k});
            expr = expr - s2{j,k}*ineq_constraints{j,1}*ineq_constraints{k,1};       
        end
    end
end
 
% % Multiply first layer by input layer
% for j = 1:size(ineq_constraints,1) 
%     if ineq_constraints{j,2}(1) == 1
%         for k = 1:dim_in
%             [prog,s3{j}] = sospolyvar(prog,monomials(vars,0:a));
%             prog = sosineq(prog,s3{j});
%             [prog,s4{j}] = sospolyvar(prog,monomials(vars,0:a));
%             prog = sosineq(prog,s4{j});
%             expr = expr - s3{j}*ineq_constraints{j,1}*con_in1(k);
%             expr = expr - s4{j}*ineq_constraints{j,1}*con_in2(k);    
%         end
%     end
% end

end


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


end