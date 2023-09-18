
function [ExpectationVectorOut , stdOut , Theta1] = computeFeatures_MultiDim(data , expectationVector , Theta0 , cdtParams)
%
% function [expectationVectorOut , stdOut] = computeFeatures_MultiDim(data , expectationVector , Theta0, cdtParams)
%
% compute the expected feature value including the incoming data
%
% input description
%
% data                  incoming observation.
%                       size(data) = [dimension of observation (d in [1]), samples of each window (nu in[1]), observation seen so far (T in [1])];
% expectationVector     vector/matrix containing the previous estimates of feature expectations
% Theta0                structure describing features in stationary contitions, obtained from computeTrainFeatures_MultiDim
% weights               The weights used in the polynomial estimator ('uniform' [default], 'parabolic' or 'constant+alpha');
%
%
% output
%
% expectationVectorOut  value of the polynomial estimator
% stdOut                standard deviation of the polynomial estimator
% Theta1                updated parameters (only when higher order interpolation is used).
%                       when other values of weights are set, Theta1 = Theta0
%                       fields concerning features computed from training set, Theta1.stdDev are NEVER updated
%
% Giacomo Boracchi
% Politecnico di Milano
% January 2010
% giacomo.boracchi@polimi.it
%
% Revision History
% October 2010 -   added LPA estimates for both features
% December 2010 -  added 1st order interpolation for sample mean
% November 2013 - added pointwise transform
% April 2014 -       Engineering, changed the way features to be computed are specified removed nubmerOfFeatures and weights, features are now specified in testParams.featuersToCompute
%

% please cite this work as
%
% [1]  Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
% "A just-in-time adaptive classification system based on the intersection of confidence intervals rule,"
% Neural Networks, Elsevier vol. 24 (2011), pp. 791-800 doi: 10.1016/j.neunet.2011.05.012
%
% [2] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
%  "Just In Time Classifiers for Recurrent Concepts"
%  Neural Networks and Learning Systems, IEEE Transactions on 2013. vol. 24, no. 4, pp. 620 - 634
%  doi:10.1109/TNNLS.2013.2239309
%
% [3] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
% "An Effective Just-in-Time Adaptive Classifier for Gradual Concept Drifts "
% in Proceedings of IJCNN 2011, the International Joint Conference on Neural Networks, San Jose, California July 31 - August 5, 2011. pp 1675 -1682,
% doi: 10.1109/IJCNN.2011.6033426
%
% [4] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
%  "A Hierarchical, Nonparametric Sequential Change-Detection Test "
%  in Proceedings of IJCNN 2011, the International Joint Conference on Neural Networks, San Jose, California
% July 31 - August 5, 2011. pp 2889 - 2896, doi: 10.1109/IJCNN.2011.6033600 %  doi:10.1109/TNNLS.2013.2239309
%
% [5] Giacomo Boracchi, Manuel Roveri
% A Reconfigurable and Element-wise ICI-based Change-Detection Test for Streaming Data
% CIVEMSA 2014, Computational Intelligence and Virtual Environments for Measurements Systems and Applications
%
%
%  References
% [Box64] Box, George EP and Cox, David R
% "An analysis of transformations",
% Journal of the Royal Statistical Society. Series B (Methodological)
% pp 211--252 1964,
%
% [Manly76] B. Manly, ?xponential data transformations,?The Statistician, pp.
% 37?2, 1976.
%
% [Mudholkar81] Mudholkar G. S., Trivedi Mom. C.: A Gaussian Approximation to the Distribution of the
% Sample Variance for Nonnormal Populations, Journal of the American Statistical
% Association, Vol. 76, No. 374 (Jun., 1981), pp. 479-485

% save the vector since it is modified only when 1st order interpolation is used
Theta1 = Theta0;



%% update the feature vector consi

% il numero di colonne di expectation vector rappresenta il numero di features values osservati finora
% the number of expectation vector columns represents the number of features values ??observed so far
nPreviousElements = size(expectationVector , 2);

for ff  = 1 : numel(Theta0)
    
    
    % this is the average over the most recent window, eventually 
    % nPreviousElements + 1 th window = most recent window 
    current_expectation_estimate = mean(data(: , nPreviousElements + 1, :), 3);
    
    % in case of Gaussian transforms to monitor the process expeectation, preliminarily transform data [Box64, Manly76]
    if any(strcmpi(Theta0(ff).featureName, {'BoxCox', 'Manly'}))
        % in principle these transforms are expected to be applied in a point-wise manner, in this case size(data, 2) = 1, even though they could be used also window-wise.
        current_expectation_estimate = mean(data(: , nPreviousElements + 1, :), 3);
        current_expectation_estimate = Theta0(ff).transform(current_expectation_estimate);
    end

    % for 2D: ff=1 = sampleMeans
    % iterative expression to compute the expectation over sample mean. It is the same for Bernoulli samples.
    if (any(strcmpi(Theta0(ff).featureName, {'sampleMeans', 'BoxCox', 'Manly', 'classificationError'})))
        mu_hat = (expectationVector(: , end , ff) * nPreviousElements + current_expectation_estimate) / (nPreviousElements + 1);
        stdEst = Theta0(ff).stdDev ./ sqrt(nPreviousElements + 1);
    end
    
    % use a LPA instead of OLS. All data have to be processed newly
   if (any(strcmpi(Theta0(ff).featureName, {'sampleMeansLPA'})))
        means = mean(data(: , 1 : nPreviousElements + 1 , :) , 3);
        [mu_hat, stdEst] = zeroOrderFitParabolicWeights(means, Theta0(ff).stdDev);
   end
    
    % polynomial fit, updates also the estimate Theta1
    if (any(strcmpi(Theta0(ff).featureName, {'polyFitOrder1'})))
        % point where the estimate is provided, we take the left-point = 0;
        [mu_hat , temp] = firstOrderFit(current_expectation_estimate, Theta0(ff));
        Theta1(ff).mu_hat = temp.mu_hat;
        Theta1(ff).stdEst = temp.stdEst;
    end
    
     % for 2D: ff=2 = gaussianizedSampleVariance
    % compute the standard deviation
    if (any(strcmpi(Theta0(ff).featureName, {'gaussianizedSampleVariance', 'gaussianizedSampleVarianceOrder1', 'gaussianizedSampleVarianceLPA'})))
        % perform detrending before computing sample variance, then use 0 order interpolation
        if (strcmpi(Theta0(ff).featureName, 'gaussianizedSampleVarianceOrder1'))
            detrendingFilter = [-1 1];
            vals = reshape(data(: , nPreviousElements + 1 , :) , [1 , length(data(: , nPreviousElements + 1 , :))]);
            detrendedVals = convn(vals , detrendingFilter , 'same');
            % downsampling sempre presente
            S2 = var(detrendedVals(1 : 2 : end));
        else
            S2 = mean((data(: , nPreviousElements + 1 , :) - repmat(current_expectation_estimate , [1 , 1 , size(data , 3)])) .^ 2 , 3);
        end
        

        if size(data , 1) == 1
            % compute the Gaussian approximating function
           
            T = Theta0(ff).transform(S2);
        elseif size(data , 1) == 2
            % compute the Gaussian approximating function
            T = Theta0(ff).transform(S2(1), S2(2));
            T = T';
        else
            for dd = 1 : size(data , 1)
                T(dd) = (S2(dd) ./ Theta0(ff).k1(dd)) .^ Theta0(ff).h0(dd);
            end
        end

        % updates the values of the sampleVariance 
        mu_hat = (expectationVector(: , end , ff)  * nPreviousElements + T') / (nPreviousElements + 1);
        stdEst = Theta0(ff).stdDev ./ sqrt(nPreviousElements + 1);
        
        if any(strcmpi(Theta0(ff).featureName, {'sampleMeansLPA'}))
            values = [expectationVector(: , end , ff) , T];
            [mu_hat , stdEst] = zeroOrderFitParabolicWeights(values , Theta0(ff).stdDev);
        end
    end
    
    % update vectors of estimate
    ExpectationVectorOut(: , : , ff) = mu_hat;
    stdOut(: , : , ff) = stdEst;
    
end

