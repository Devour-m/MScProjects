function [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval(BoundTable,P_hat)
%% this function is used to retrieve bound values for tpr, tnr, ppv and npv given estimated P_hat
% from BoundTable
% BoundValue = [detect_lb;warn_lb;warn_ub;detect_ub];
%%
temp = round(P_hat/0.001); % retrieve temp value in bound table
if temp>=999
    temp = 999;
elseif temp<=1
    temp = 1;
end

BoundValue = BoundTable(:,temp);

detect_lb = BoundValue(1);
warn_lb = BoundValue(2);
warn_ub = BoundValue(3);
detect_ub = BoundValue(4);

end