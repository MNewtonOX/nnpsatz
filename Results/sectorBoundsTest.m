% Find optimal position for sector bounds

% Start off with numerical method to get rough idea
clear 
clc

j = 1;
for x_l = -10:0.25:-1 %-10:0.25:1
for x_u = 1:0.25:10 %1:0.25:10 %(0+0.1):1:(10+0.1)

%x_l = -5;
%x_u = 5;
k = 1;
tic
for x_ml = (x_l+0.01):(abs(x_l)/10):(0)
    for x_mu = (0):(abs(x_u)/10):(x_u-0.01)
        
        % Find line L_ub 
        syms d1
        d1 = vpasolve(sig(d1)*(1 - sig(d1)) == (sig(x_mu) - sig(d1))/(x_mu - d1),d1,[-10,0]);
        grad_L_ub = sig(d1)*(1 - sig(d1));
        c_L_ub = sig(d1) - grad_L_ub*d1;
        %grad_L_ub = 1/4;
        %c_L_ub = sig(x_mu) - grad_L_ub*x_mu;
        p1 = (sig(x_u) - sig(x_mu))/grad_L_ub + x_mu; 
        
        % Find line L_ua
        grad_L_ua = (sig(x_mu) - sig(x_u))/(x_mu - x_u);
        c_L_ua = sig(x_u) - grad_L_ua*x_u;
        
        % Fine line L_lb
        syms d2
        d2 = vpasolve(sig(d2)*(1 - sig(d2)) == (sig(x_ml) - sig(d2))/(x_ml - d2),d2,[0,10]);
        grad_L_lb = sig(d2)*(1 - sig(d2));
        c_L_lb = sig(d2) - grad_L_lb*d2;
        %grad_L_lb = 1/4;
        %c_L_lb = sig(x_ml) - grad_L_lb*x_ml;
        p2 = (sig(x_l) - sig(x_ml))/grad_L_lb + x_ml;
        
        % Find line L_la
        grad_L_la = (sig(x_ml) - sig(x_l))/(x_ml - x_l);
        c_L_la = sig(x_l) - grad_L_la*x_l;
        
        % Find intersection of lines p3
        %syms p3
        %p3 = solve(c_L_ua + grad_L_ua*p3 == c_L_lb + grad_L_lb*p3);
        p3 = (c_L_lb - c_L_ua)/(grad_L_ua - grad_L_lb);
        
        % Find intersection of lines p4
        %syms p4
        %p4 = solve(c_L_la + grad_L_la*p4 == c_L_ub + grad_L_ub*p4);
        p4 = (c_L_ub - c_L_la)/(grad_L_la - grad_L_ub);

        areaA = 0.5*(sig(x_u) - sig(x_mu))*(x_u - p1);
        areaC = 0.5*(sig(x_ml) - sig(x_l))*(p2 - x_l);
        
        syms x
        areaB = int((grad_L_lb - grad_L_la)*x + c_L_lb - c_L_la, x_ml, p4);
        areaB = areaB + int((grad_L_lb - grad_L_ub)*x + c_L_lb - c_L_ub, p4, p3);
        areaB = areaB + int((grad_L_ua - grad_L_ub)*x + c_L_ua - c_L_ub, p3, x_mu);
        
        areaT(k) = areaA + areaB + areaC; 
        coord(k,1) = x_ml;
        coord(k,2) = x_mu;
        k = k + 1;
        %coord(k,3) = x_l;
        %coord(k,4) = x_u;
        

        
%         fplot(1/(1+exp(-x)))
%         hold on
%         fplot(grad_L_ub*x + c_L_ub)
%         fplot(grad_L_ua*x + c_L_ua)
%         
%         fplot(grad_L_lb*x + c_L_lb)
%         fplot(grad_L_la*x + c_L_la)
    end
end
toc
bounds(j,1) = x_l;
bounds(j,2) = x_u;
min_area(j) = min(areaT)
min_index(j) = find(areaT == min_area(j))
min_coord(j,:) = coord(min_index(j),:)

j = j + 1;

end
end

resultsValues.bounds = bounds;
resultsValues.minArea = min_area;
resultsValues.minCoord = min_coord;

%%       
%guess1 = x_u*sig(x_u + x_l) +2*sig(x_u - 3)*exp(-(x_u + x_l)^2)
%guess2 = x_u*sig(x_u + x_l) - (x_u/2 - 2)*exp(-(x_u + x_l)^2)

% For some reason this works
%guess3 = x_u*sig(x_u + x_l) + (2*sig(x_u - exp(1)) - x_u/2)*exp(-(x_u + x_l)^2)

%guess4 = x_l*sig(-x_u - x_l) - (2*sig(-x_l - exp(1)) + x_l/2)*exp(-(x_u + x_l)^2)
%end
%end

% test1 = min(areaT)
% test2 = find(areaT == test1)
% coord(test2,:)
% 
% k = 1;
% areaT2 = zeros(size(areaT));
% for j = 1:size(areaT,2)
%     if areaT(j) >= 0 
%         if areaT(j) <= 100
%             areaT2(k) = areaT(j); k = k + 1;
%         end
%     end
% end

% k = 1;
% for x_ml = (x_l+0.01):0.1:(0-0.01)
%     for x_mu = (0+0.01):0.1:(x_u-0.01)
%         
%         % Find line L_ub 
%         syms d1
%         d1 = solve(sig(x_mu) - sig(d1) == sig(d1)*(1 - sig(d1))*(x_mu - d1));
%         grad_L_ub = (sig(x_mu) - sig(d1))/(x_mu - d1);
%         c_L_ub = sig(x_mu) - grad_L_ub*x_mu;
%         p1 = (sig(x_u) - sig(x_mu))/grad_L_ub + x_mu; 
%         
%         % Find line L_ua
%         grad_L_ua = (sig(x_mu) - sig(x_u))/(x_mu - x_u);
%         c_L_ua = sig(x_u) - grad_L_ua*x_u;
%         
%         % Fine line L_lb
%         syms d2
%         d2 = solve([(sig(x_ml) - sig(d2))/(x_ml - d2) == sig(d2)*(1 - sig(d2))]);
%         grad_L_lb = (sig(x_ml) - sig(d2))/(x_ml - d2);
%         c_L_lb = sig(x_ml) - grad_L_lb*x_ml;
%         p2 = (sig(x_l) - sig(x_mu))/grad_L_lb + x_ml;
%         
%         % Find line L_la
%         grad_L_la = (sig(x_ml) - sig(x_l))/(x_ml - x_l);
%         c_L_la = sig(x_l) - grad_L_la*x_l;
%         
%         % Find intersection of lines p3
%         syms p3
%         p3 = solve(c_L_ua + grad_L_ua*p3 == c_L_lb + grad_L_lb*p3);
%         
%         % Find intersection of lines p4
%         syms p4
%         p4 = solve(c_L_la + grad_L_la*p4 == c_L_ub + grad_L_ub*p4);
% 
%         areaA = 0.5*(sig(x_u) - sig(x_mu))*(x_u - p1);
%         areaC = 0.5*(sig(x_ml) - sig(x_l))*(p2 - x_l);
%         
%         syms x
%         areaB = int((grad_L_lb - grad_L_la)*x + c_L_lb - c_L_la, x_ml, p4);
%         areaB = areaB + int((grad_L_lb - grad_L_ub)*x + c_L_lb - c_L_ub, p4, p3);
%         areaB = areaB + int((grad_L_ua - grad_L_ub)*x + c_L_ua - c_L_ub, p3, x_mu);
%         
%         areaT(k) = areaA + areaB + areaC; k = k + 1;
%         
% %         fplot(1/(1+exp(-x)))
% %         hold on
% %         fplot(grad_L_ub*x + c_L_ub)
% %         fplot(grad_L_ua*x + c_L_ua)
% %         
% %         fplot(grad_L_lb*x + c_L_lb)
% %         fplot(grad_L_la*x + c_L_la)
%     end
% end
% 
% k = 1;
% areaT2 = zeros(size(areaT));
% for j = 1:size(areaT,2)
%     if areaT(j) >= 0 
%         if areaT(j) <= 100
%             areaT2(k) = areaT(j); k = k + 1;
%         end
%     end
% end

function y = sig(x)
    y = 1/(1 + exp(-x));
end