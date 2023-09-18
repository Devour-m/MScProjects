function  [refined_val , testDetectingChange , midpoint] = f_ICI_refine_test(DataSetComplete, detected, dimensionDetected, tsLength, testParams, Theta0)
%
% function  [refined , testDetectingChange , midpoint] = f_ICI_refine_test(DataSetComplete, detected, dimensionDetected , tsLength , testParams, Theta0)
%
% Implement the refinement procedure presented in 
%
% [1] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
% "A just-in-time adaptive classification system based on the intersection of confidence intervals rule"
%  Neural Networks, Elsevier vol. 24 (2011), pp. 791-800 (doi: 10.1016/j.neunet.2011.05.012)
%
% [2] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
% "Adaptive Classifiers with ICI-based Adaptive Knowledge Base Management"
% ICANN 2010, 20th International Conference on Artificial Neural Networks, September 15-18, 2010, Thessaloniki, Greece,
% Lecture Notes on Computer Science. Springer Berlin / Heidelberg, vol 6353, pg 458-467
%
% Revision History
% April 2010              first release
% December 2013     added Theta0 as an optional parameter to avoid multiple calls to computeTrainingFeatures()
%
% Giacomo Boracchi
% Politecnico di Milano
% giacomo.boracchi@polimi.it

% if it does not exist passes an empty variable -> implies call to computeTrainingFeatures
if (~ exist('Theta0' , 'var'))
    Theta0 = [];
end

if (~exist('Gamma' , 'var') || isempty(Gamma))
    Gamma = testParams.Gamma;
end

%% 
K = testParams.K_refinement;
windowWidth = testParams.windowWidth;


% modified by Shuyi dimensionDetected ---> :

% if the data set has more than one feature, limit the refinement to the
% detected dimension 
if size(DataSetComplete , 1) > 1
    % DataSet = DataSetComplete(:,1 : detected(end));
     DataSet = DataSetComplete(dimensionDetected, 1 : detected(end));
else
    DataSet = DataSetComplete(1 : detected(end));
end



% Shuyi Zhang 2020
TrainingSet = DataSet(1 : tsLength);
if (~exist('Theta0' , 'var') || isempty(Theta0))
    do_computeTrainingFeatures = 1;
else
    do_computeTrainingFeatures = 0;
end

% relearn Theta0 for the dimension where a change is detected 
% for multi-dimensional data, the previously learnt Theta0 does not have a
% transform function 
% only dimension=2 and dimension=1 have transform function 
if do_computeTrainingFeatures
    Theta0 = computeTrainFeatures_MultiDim(TrainingSet, testParams, Gamma);
    % numel(Theta0) = 2 even for multidim 
end


split = 1;
s = 1;
% la lunghezza ?pari alla met?del validation set (lunghezza DataSet - trainingSet)

% take the first midpoint
midpoint = floor((length(DataSet) - tsLength) / K) + tsLength;
%'here'
%length(DataSet)
% midpoint
% 413
% initialize the refinement in case there won't be any detection of the dataset
refined = detected;
refined_tmp = [];

while (split)
	% define the dataset to be analyzed
    ValidationSet_temp = DataSet(:,midpoint(s) + 1: end);
    % size 7
    %size(DataSet)
    %size(ValidationSet_temp)
    %length(ValidationSet_temp)
    % 54???
    roundedValidationSetLength = floor(size(ValidationSet_temp,2) / windowWidth) * windowWidth;
    % roundedValidationSetLength 
    % 40
    ValidationSet = ValidationSet_temp(:,end - roundedValidationSetLength +1 : end);
    TrainingSet = DataSet(:,1 : tsLength);
    DataSetReduced = [TrainingSet , ValidationSet];
    
    
	% this is used only for cdt on linear trends, perform detrending
    if any(strcmpi(testParams.featuresToCompute , 'polyFitOrder1'))
        % Detrending from the Dataset
        % estimate the regression line from the midpoint
        validationSetStartsAt = size(DataSet,2) - size(ValidationSet,2);
        regParams = polyfit([0 : validationSetStartsAt - 1] , DataSet(1 : validationSetStartsAt) , 1);
        midpointVal = regParams(1) * (validationSetStartsAt - 1) + regParams(2);
        tsVal =  regParams(1) * (tsLength - 1) + regParams(2);
        % align the two segments to avoid abrupt changes in between
        ValidationSet = ValidationSet - midpointVal + tsVal;
        DataSetReduced = [TrainingSet , ValidationSet];
    end

	% run the CDT on th reduced dataset, specifying the GammaRefinement parameter to be used
    [refined_tmp(s) , testDetectingChange(s) , Theta0 , dimensionWhereChangeDetected(s)] = f_ICI_test_MultiDim(DataSetReduced, tsLength, testParams, testParams.GammaRefinement, Theta0);
    
%    tsLength
%    DataSetReduced(end-10:end)
%    size(DataSetReduced)
%    'wtf'
%    testParams
%    testParams.GammaRefinement
%    Theta0
 
    % When the refinement does not provide any detection we trust the previous one, which probably was close enough
    if refined_tmp(s) == 0
        % 'no change detected in the refinement procedure -> set T_hat'
        refined(s) = detected(end);
        
    else
        % update refined position
        refined(s) = midpoint(s) + refined_tmp(s) - tsLength;
	end
    
    s = s + 1;
    
    % update the midpoint position
    midpoint(s) = floor(size(ValidationSet,2) /  K) + midpoint(s-1);
    
    % Check if the split has not reached the end of the dataset or if the next split point has exceeded one of the estimates obtained during the refinement procedure
    if(min(refined) <= midpoint(s) || midpoint(s) == midpoint(s-1))
        split = 0;
    end
    
end

% return value
refined_val = min(refined);