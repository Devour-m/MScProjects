function [h_vals , p_vals , X1 , X2] = validateChanges_Data(inputDataSet, featureDetectingChange , oldTS_Init , oldTS_End , newTS_Init , newTS_End , params)
%
% function [h_vals , p_vals] = validateChanges_Data(DataSet , featureDetectingChange , oldTS_Init , oldTS_End , newTS_Init , newTS_End , params)
%
% Performs validation of a change detected by the ICI-based CDT using the Hotelling T-square Statistic
% features are extracted from DataSet(oldTS_Init : oldTS_End), considering params (parameter structure for the ICI_basd CDT)
%
% Hotelling T^2 statistic is used to assess if the feature mean over DataSet(oldTS_Init : oldTS_End) equals those computed over DataSet(newTS_Init : newTS_End)
% IMPORTANT NOTE:
%       When there is more than one sensor, stack in each row the vector, the values of the same feature from a specific sensor
%       When there is a single sensor, then stack in each row of the vector different features.
%
% cft "Applied Multivariate Statisitcal Analysis" Johnson Wichern Cap 5 for tests on Multivariate Normals
%
% please cite this work as:
% [1] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
% "A Hierarchical, Nonparametric Sequential Change-Detection Test"
%  in Proceedings of IJCNN 2011, the International Joint Conference on Neural Networks,
% San Jose, California July 31 - August 5, 2011. pp 2889 - 2896, DOI: 10.1109/IJCNN.2011.6033600
%
% [2] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
% "A just-in-time adaptive classification system based on the intersection of confidence intervals rule"
%  Neural Networks, Elsevier vol. 24 (2011), pp. 791-800 (doi: 10.1016/j.neunet.2011.05.012)
%
%
% Giacomo Boracchi
% Politecnico di Milano
% December 2011
% giacomo.boracchi@polimi.it
%
% Revision History
% December 2011 -   First Release, taken out of f_ICI_test_MultiChange_Distributed




%% modified by Shuyi
% original was 
%if(iscell(DataSet) == 0)
%        temp{i} = DataSet;
%        DataSet{i} = temp;
%        clear temp
%end
%

if(iscell(inputDataSet) == 0)
    for i = 1:size(inputDataSet,1)
        temp = inputDataSet(i,:);
        DataSet{i} = temp;
        clear temp
    end
end

X1 = [];
X2 = [];

MM1 = [];
MM2 = [];
VV1 = [];
VV2 = [];

N_SENSORS = numel(DataSet);


% compute *ALL the features* on *ALL the sensors*
for b = 1 : N_SENSORS
    
    % questo lo salvo in una cella semplicemente per gestire quando modifico tutti i DataSet
    oldTS{b} = DataSet{b}(: , oldTS_Init : oldTS_End);
    newTS = DataSet{b}(: , newTS_Init : newTS_End);
    
    %  determine the gaussianizing transform for the sample variance
    [Theta] = computeTrainFeatures_MultiDim(oldTS{b}, params, params.Gamma);
            
    reshaped_oldTS = [];
    for ii = 1 : (length(oldTS{b}) / params.windowWidth)
        reshaped_oldTS(:,ii,:) = oldTS{b}(: , (ii - 1) * params.windowWidth + 1 : ii * params.windowWidth);
    end
    
    reshaped_newTS = [];
    for ii = 1 : (length(newTS) / params.windowWidth)
        reshaped_newTS(:,ii,:) = newTS(: , (ii - 1) * params.windowWidth + 1 : ii * params.windowWidth);
    end
    
    % first feature: the sample mean, compute the mean
    M1 = mean(reshaped_oldTS , 3);
    M2 = mean(reshaped_newTS , 3);
    
    % second feature: the transformed sample variance
    
    % squared errors
    SE1 = (reshaped_oldTS - repmat(M1 , [1 , 1 , params.windowWidth])) .^ 2;
    SE2 = (reshaped_newTS - repmat(M2 , [1 , 1 , params.windowWidth])) .^ 2;
    
    % variances
    S1 = mean(SE1 ,  3);
    S2 = mean(SE2 ,  3);
    
    
    %% 
    % transformed variance
    V1 = (S1 / Theta(2).k1) .^ Theta(2).h0;
    V2 = (S2 / Theta(2).k1) .^ Theta(2).h0;
    
    % send the data to the central unit in order to compute the S_pooled
    MM1(b , :) = M1;
    MM2(b , :) = M2;
    
    VV1(b , :) = V1;
    VV2(b , :) = V2;
    
end % compute features out of sensors

%% Centralized T^2 hotelling test, run hypothesis test on a *SINGLE feature, ALL the sensors*

% if there is more than one sensor, stack in each row the vector, the values of the same feature from a specific sensor
if (N_SENSORS > 1)
    % Chose the feature
    if (featureDetectingChange == 1)
        % change on the sample mean
        X1 = MM1;
        X2 = MM2;
    else % chanage on the variance
        if (featureDetectingChange == 2)
            X1 = VV1;
            X2 = VV2;
        else
            disp(['W.A.F selected feature is ' , num2str(featureDetectingChange)]);
        end
    end
    % Test Dimension
    p = N_SENSORS;
    
else % if there is a single sensor, then stack in each row of the vector different features.
    
    X1 = [MM1 ; VV1];
    X2 = [MM2 ; VV2];
    p = 2;
    
end

% H0
delta  = zeros(p , 1);

if isempty(newTS) == 0
    [h_vals , p_vals] = HotellingTSquareTest(X1 , X2 , delta , params.alpha);
else
    h_vals = 0;
    p_vals = [];
end
