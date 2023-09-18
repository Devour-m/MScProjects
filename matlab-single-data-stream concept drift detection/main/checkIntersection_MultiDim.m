function [res, testDetectingChange,dimensionWhereChangeDetected] = checkIntersection_MultiDim(intersection_interval_UP,intersection_interval_DWN,interval_UP,interval_DWN)
%
%  [res, testDetectingChange,dimensionWhereChangeDetected] = checkIntersection_MultiDim(intersection_interval_UP,intersection_interval_DWN,interval_UP,interval_DWN)
% 
% Check if the multivariate interval [interval_UP,interval_DWN] intersects [intersection_interval_UP,intersection_interval_DWN]
%
% returns res=1 iff there is a non empty intesection in all featuers testDetectingChange indicates the test that failed
%
% Giacomo Boracchi
% Politecnico di Milano
% January 2010
% giacomo.boracchi@polimi.it
%

testDetectingChange = 0;
dimensionWhereChangeDetected = 0;

upperIntersection = (interval_DWN < intersection_interval_UP);
lowerIntersection = (interval_UP > intersection_interval_DWN);

% returns
res = all(all(upperIntersection .* lowerIntersection));

%change is detected, save the test and the dimension where the change has
%been detected
if res == 0
    [dimensionWhereChangeDetected , testDetectingChange] = find(upperIntersection .* lowerIntersection == 0);
end