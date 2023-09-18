function [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval_2d(BoundTable,P_hat,eta_hat)
%% this function is used to retrieve bound values for tpr, tnr, ppv and npv 
%% given estimated P_hat and eta_hat
% from BoundTable
% BoundValue = [detect_lb;warn_lb;warn_ub;detect_ub];
%%
temp1 = round(P_hat/0.001); % retrieve temp value in bound table for p_hat
if temp1>=999
    temp1 = 999;
elseif temp1<=1
    temp1 = 1;
end

temp2 = round(eta_hat/0.001)-800; % retrieve temp value in bound table for eta_hat
if temp2>=199
    temp2 = 199;
elseif temp2<=1
    temp2 = 1;
end

BoundValue = BoundTable(:,temp1,temp2);

detect_lb = BoundValue(1);
warn_lb = BoundValue(2);
warn_ub = BoundValue(3);
detect_ub = BoundValue(4);

end