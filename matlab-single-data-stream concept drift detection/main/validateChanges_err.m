function [res , p_val , p_hat_1 , p_hat_2] = validateChanges_err(class_error, oldTS_Init , oldTS_End , newTS_Init , newTS_End, alpha , do_reverse_test)
%
% function [res , p_val] = validateChanges_err(class_error, oldTS_Init , oldTS_End , newTS_Init , newTS_End , alpha , do_reverse_test)
%
%Test difference between proportions having unequal size: used to validate changes in the Classification Error detected by the
% ICI-based CDT meant for Bernoulli sequences.
% H0 : p_hat_1  == p_hat_2
%
% input description
% Test equal proportions of successes over class_error(oldTS_Init : oldTS_End) and class_error(newTS_Init : newTS_End)
% by running an hypothesis test of level alpha
%
% ouput
% res == 1 the null hypothesis H0 is rejected, change is validated
% res == 0 the null hypothesis H0 cannot be rejected, change is not validated
%
%
% cft 10-6 INFERENCE ON TWO POPULATION PROPORTIONS. Montgomery
%
% Giacomo Boracchi
% Politecnico di Milano
% December 2011
% giacomo.boracchi@polimi.it
%
% Revision History
% December 2011 -   First Release, taken out of f_ICI_test_MultiChange_Distributed


if (~exist('do_reverse_test' , 'var'))
    do_reverse_test = 0;
end

n_1 = oldTS_End - oldTS_Init + 1;
p_hat_1  = sum(class_error(oldTS_Init : oldTS_End)) / n_1;
var_1 = p_hat_1 * (1 - p_hat_1) / n_1;

n_2 = newTS_End - newTS_Init;
p_hat_2 = sum(class_error(newTS_Init : newTS_End)) / n_2;
var_2 = p_hat_2 * (1 - p_hat_2)  / n_2;

p_hat = mean([class_error(oldTS_Init : oldTS_End) , class_error(newTS_Init : newTS_End)]);
std_hat = sqrt(p_hat * (1 - p_hat) * (1 / n_1 + 1 / n_2));

if (do_reverse_test == 0)
    % sotto H0 : p_hat_1  == p_hat_2, T_stat è distributito come una normale N(0 , 1)
    test_statistic = (p_hat_1 - p_hat_2) / std_hat;
    Q_norm_1 = norminv(alpha / 2 , 0 , 1);
    Q_norm_2 = norminv(1 - alpha / 2 , 0 , 1);
    
    % res == 0 se valido il cambiamento, res == 1 se non lo valido
    res  = (test_statistic < Q_norm_1) ||  (test_statistic > (Q_norm_2));
else
%      % H0 : |p_hat_1  - p_hat_2 | >  do_reverse_test * std_hat,
%      test_statistic = (p_hat_1 - p_hat_2 - do_reverse_test * std_hat ) / std_hat;
%      % sotto H0 test_statistic è distribuito come una N(0,1)
% 
%      Q_norm_1 = norminv(alpha / 2 , 0 , 1);
%     Q_norm_2 = norminv(1 - alpha / 2 , 0 , 1);
%      
%      res  = (test_statistic < Q_norm_1) ||  (test_statistic > (Q_norm_2));
end
p_val = 2 * (1 - normcdf(abs(test_statistic)));