function [Theta] = computeTrainFeatures_MultiDim(trainingSet, params, Gamma)
%
% function [Theta] = computeTrainFeatures_MultiDim(trainingSet, params, Gamma)
%
% compute features to configure the test out of the trainingSet
%
% Gamma            - which is possibly modified and returned as a field of Theta to the function when do_adaptiveGamma == 1
% weights          - The weights used in the polynomial estimator ('uniform' [default], 'parabolic' or 'constant+alpha');
% do_adaptiveGamma - default 0
%
% output
%
% Theta structure array its fieds are
%
% Theta(i).expect       expected value of the i-th feature
% Theta(i).stdDev       standard deviation of the i-th feature on the training set. IT IS NOT RELATED TO THE ESTIMATE. is \sigma
% Theta(i).function     Transofmation that has to be applied of the i-th feature on the training set
% Theta(i).Gamma        used only when do_adaptiveGamma == 1
%
%  this hold for features 1 and 2 (sample mean and sample variance of the process)
%  feature 2 has additional fields as the parameter h0
%
% Giacomo Boracchi
% Politecnico di Milano
% January 2010
% giacomo.boracchi@polimi.it
%
% Revision History
% October 2010 -  added LPA estimates for both features
% December 2010 - added 1st order interpolation for sample mean
% December 2010 - added support for vectors of Gamma
% November 2013 - added pointwise transform
% April 2014 -       Engineering, changed the way features to be computed are specified removed nubmerOfFeatures and weights, features are now specified in testParams.featuersToCompute
%
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
% [Manly76] B. Manly, �Exponential data transformations,� The Statistician, pp.
% 37�42, 1976.
%
% [Mudholkar81] Mudholkar G. S., Trivedi Mom. C.: A Gaussian Approximation to the Distribution of the
% Sample Variance for Nonnormal Populations, Journal of the American Statistical
% Association, Vol. 76, No. 374 (Jun., 1981), pp. 479-485
%

errorFrom = 'computeTrainFeatures_MultiDim';

windowWidth = params.windowWidth;

if ~exist('Gamma', 'var')
    Gamma = params.Gamma;
end

do_adaptiveGamma = params.do_adaptiveGamma;

% reshaped=reshape(trainingSet,[size(trainingSet,1),(length(trainingSet))/windowWidth,windowWidth]);

% i dati sono organizzati nel seguente modo:
% length(trainingSet)---> size(trainingSet,2)
% 以前是length(trainingSet)也就是说handle不了dimension 大于windowwidth的数据流
% 2020
for ii = 1 : (size(trainingSet,2) / windowWidth)
    reshaped(:,ii,:) = trainingSet(:, (ii - 1) * windowWidth + 1 : ii * windowWidth); % nFeature * nWindow * windowWidth
    %size(reshaped)
end

do_downsamplig = 0;

%% first sample moment of X - we have to report the value of noise standard deviation, not the st

M = mean(reshaped, 3); % vettore delle medie locali
sigma_noise = std(M')';

for ff  = 1 : numel(params.featuresToCompute)
    
    % sample mean as in [1,2,4]
    if strcmpi(params.featuresToCompute{ff}, 'sampleMeans')
        mu_hat = mean(M, 2);
        
        Theta(ff).expect = mu_hat;
        Theta(ff).stdDev = sigma_noise;
        Theta(ff).transform = @(x)(x);
    end
    
    % sample mean using LPA instead of OLS as in [1]
    if strcmpi(params.featuresToCompute{ff}, 'sampleMeansLPA')
        [mu_hat, sigmaEst] = zeroOrderFitParabolicWeights(M, sigma_noise);
        Theta(ff).expect = mu_hat;
        Theta(ff).stdDev = sigmaEst;
        Theta(ff).transform = @(x)(x);
    end
    
    % detrended sample mean by first order polynomial regression, as in [4]
    if strcmpi(params.featuresToCompute{ff}, 'polyFitOrder1')
        % point where the estimate is provided, we take the left-
        point = 0;
        [mu_hat, Theta0] = firstOrderFit(M, point);
        
        % initialize the Theta parameter (copia mu_hat = Theta0.mu_hat, sigma_noise = Theta0.sigma_noise)
        Theta(ff) = Theta0;
        Theta(ff).transform = @(x)(x);
    end
    
    % compute the expected success rate p by averaging the mean of windowWidth bernulli i.i.d. trials as in [2] meant for Observations that are Bernoulli and i.i.d.
    if strcmpi(params.featuresToCompute{ff}, 'classificationError')
        mu_hat = mean(M, 2);    % parametro della Bernulliana
        varianceEst = (mu_hat * (1 - mu_hat)); % da definizione della Bernulliana
        sigma_noise = sqrt(varianceEst / windowWidth); %  la varianza della Binomiale � \nu p (1 - p), la deviazione standard della media di Bernulliane � quindi questa
        
        Theta(ff).expect = mu_hat;
        Theta(ff).stdDev = sigma_noise;
        Theta(ff).transform = @(x)(x);
    end
    
    % compute the powerlaw coefficient according to  [Box64] as in [5]
    if strcmpi(params.featuresToCompute{ff}, 'BoxCox')
        
        % we take M as the training set for learning the Gaussian tranform. When params.windowWidth == 1 this is the
        % traioning set, otherwise the tranform can be applied on average over nonoverlapping data windows as well
        trainingSet = M;
        % Making the DataSet positive for the Box-Cox transformation
        minTrainingSet = abs(min(trainingSet));
        positive_trainingSet = trainingSet + minTrainingSet  + params.pointwise.BoxCoxOffset;
        
        % Estimate Lambda
        [~, estimated_Lambda] = boxcox(positive_trainingSet');
        
        % Transform the dataset according to Box-Cox transformation and the estimated Lambda
        if estimated_Lambda ~= 0
            Theta(ff).transform = @(x)(((x + minTrainingSet + params.pointwise.BoxCoxOffset) .^ estimated_Lambda - 1) ./ estimated_Lambda);
        else
            Theta(ff).transform = @(x)(log(x + minTrainingSet + params.pointwise.BoxCoxOffset));
        end
        % save the parameters for diagnosis purpose
        Theta(ff).minTrainingSet = minTrainingSet;
        
        % save the parameters for diagnosis purpose
        Theta(ff).estimated_Lambda = estimated_Lambda;
        
        % transform the training set and compute the p-value of the JB test as an indicator of how successfull normalization
        % was
        transformedTrainingSet = Theta(ff).transform(trainingSet);
        
        % try
        %     check if a toolbox is available
        % [~, jbTest_pValue] = jbtest(trainingSet);
        % Theta(1).jbTest_pValue = jbTest_pValue;
        % end
        
        mu_hat = mean(transformedTrainingSet);
        sigma_noise = std(transformedTrainingSet);
        
        Theta(ff).expect = mu_hat;
        Theta(ff).stdDev = sigma_noise;
        
    end
    
    % compute the parameter of the exponential transform according to [Manly76]  as in [5] -- requires call to R --
    if strcmpi(params.featuresToCompute{ff}, 'Manly')
        
        % we take M as the training set for learning the Gaussian tranform. When params.windowWidth == 1 this is the
        % traioning set, otherwise the tranform can be applied on average over nonoverlapping data windows as well
        trainingSet = M;
        
        % define a random prefix to be assigned to the file name (both dataset and output)
        callIdentifier = num2str(round(1e10 *rand(1)));
        save([callIdentifier, '.txt'], 'trainingSet', '-ascii');
        
        % determine the current platform to use the proper system call
        currentPlatform = computer;
        
        % call R.... on the script  f_computeManlyTransformCoefficient.r which is located in the path, passing as
        % argument callIdentifier
        switch currentPlatform(1 : 3)
            case {'PCW','GLN'}
                %[successfulRCall] = system(['R CMD BATCH "', which('computeManlyTransformCoefficient.r'), '" Rdisplay.txt']);
                [successfulRCall] = system(['R --vanilla --slave --args "', callIdentifier, '" <"', which('computeManlyTransformCoefficient.r'),'"> ', callIdentifier, '_Rout.txt']);
            case 'MAC'
                [successfulRCall] = system(['unset DYLD_LIBRARY_PATH; /usr/bin/Rscript "', which('computeManlyTransformCoefficient.r'), '" Rdisplay.txt' ]);
        end
        
        % errors
        if successfulRCall ~= 0
            error(['error from ', errorFrom, ' system call to R did not worked']);
        end
        
        % Load estimated lambda from file
        estimated_Lambda = load([callIdentifier,'lambda.txt']);
        
        % remove files
        delete([callIdentifier,'*.txt']);
        
        % Transform the dataset according to Box-Cox transformation and the estimated Lambda
        if estimated_Lambda ~= 0
            Theta(ff).transform = @(x)((exp(estimated_Lambda * x) - 1) / estimated_Lambda);
        else
            Theta(ff).transform = @(x)(x);
        end
        
        % save the parameters for diagnosis purpose
        Theta(ff).estimated_Lambda = estimated_Lambda;
        
        % transform the training set and compute the p-value of the JB test as an indicator of how successfull normalization
        % was
        transformedTrainingSet = Theta(ff).transform(trainingSet);
        
        % try
        %     check if a toolbox is available
        % [~, jbTest_pValue] = jbtest(trainingSet);
        % Theta(1).jbTest_pValue = jbTest_pValue;
        % end
        
        mu_hat = mean(transformedTrainingSet);
        sigma_noise = std(transformedTrainingSet);
        
        Theta(ff).expect = mu_hat;
        Theta(ff).stdDev = sigma_noise;
    end
    
    % compute the Gaussianized sample variance according to [Mudholkar81] as in [1]
    if strcmpi(params.featuresToCompute{ff}, 'gaussianizedSampleVariance') || strcmpi(params.featuresToCompute{ff}, 'gaussianizedSampleVarianceOrder1') || strcmpi(params.featuresToCompute{ff}, 'gaussianizedSampleVarianceLPA')
        % When first order fitting of the is needed we have to detrend the data before computing the variance
        if strcmpi(params.featuresToCompute{ff}, 'gaussianizedSampleVarianceOrder1')
            detrendingKernel = zeros(1, 1, 2);
            detrendingKernel(1, 1, 1) = -1;
            detrendingKernel(1, 1, 2) = 1;
            reshaped = convn(reshaped, detrendingKernel, 'valid');
            meanReshaped = mean(reshaped, 3);
            SE = (reshaped - repmat(meanReshaped, [1, 1, size(reshaped, 3)])) .^ 2;
            % downsampling
            if do_downsamplig
                SE(:, 1 : 2 : end, :) = [];
            end
        else
            % calcola gli scarti quadratici dalla media per ogni elemento nel cubotto
            SE = (reshaped - repmat(M, [1, 1, windowWidth])) .^ 2;
        end
        
        % sulla terza dimensione, quindi a blocchi di windowWidth, calcola la media degli
        % scarti
        S2 = mean(SE, 3);
        
        % compute the Gaussianizing Transform according to [Mudholkar81]
        
        %compute the first 6 raw Moments vector. Il numero di righe indica il
        %momento, il numero di colonne indica la dimensione del dato analizzato
        for cc = 1 : 6
            Mom(:, cc) = mean((trainingSet) .^ cc, 2);
        end
        
        % compute the cumulants of original distribution.
        % i cumulants hanno dimensione dimensioneDeiDati x indice del cumulants
        C(:, 1) = Mom(:, 1);
        C(:, 2) = Mom(:, 2) - Mom(:, 1).^2;
        C(:, 3) = 2 * Mom(:, 1).^3 - 3 * Mom(:, 1) .* Mom(:, 2) + Mom(:, 3);
        C(:, 4) = -6 * Mom(:, 1).^4 + 12 * (Mom(:, 1).^2) .* Mom(:, 2) - 3 * Mom(:, 2).^2 - 4 * Mom(:, 1) .* Mom(:, 3) + Mom(:, 4);
        C(:, 5) = 24 * Mom(:, 1).^5 - 60 * Mom(:, 1).^3 .* Mom(:, 2) + 20 * (Mom(:, 1).^2 ) .* Mom(:, 3) - 10 * Mom(:, 2) .* Mom(:, 3) + 5 * Mom(:, 1) .* (6 * Mom(:, 2).^2  - Mom(:, 4)) + Mom(:, 5);
        C(:, 6) = -120 * Mom(:, 1).^6 + 360 * (Mom(:, 1).^4) .* Mom(:, 2) - 270* (Mom(:, 1).^2) .* Mom(:, 2).^2 + 30 * Mom(:, 2).^3 - 120 * (Mom(:, 1).^3) .* Mom(:, 3) + 120 * Mom(:, 1) .* Mom(:, 2) .* Mom(:, 3);
        C(:, 6) = C(:, 6) - 10 * Mom(:, 3).^2 + 30 * (Mom(:, 1).^2) .* Mom(:, 4)  - 15 * Mom(:, 2) .* Mom(:, 4) - 6 * Mom(:, 1) .* Mom(:, 5) + Mom(:, 6);
        
        % compute sample variance moments
        % i sample variance moments hanno dimensione dimensioneDeiDati x 1
        k1 = (windowWidth - 1) .* ones(size(C(:, 1)));
        k2 = (windowWidth - 1).^(2) .* (C(:, 4) ./ (windowWidth .* C(:, 2).^2) + 2 ./ (windowWidth-1));
        k3 = (windowWidth - 1).^(3) .* (C(:, 6) ./ windowWidth.^2 + 12 .* C(:, 4) .* C(:, 2) ./ (windowWidth * (windowWidth-1)) + (4 * (windowWidth - 2) .* C(:, 3).^(2)) ./ (windowWidth * (windowWidth-1).^2) + 8 .* (C(:, 2).^3) ./ (windowWidth - 1).^2 ) ./ C(:, 2).^3;
        
        % compute h0
        h0 = 1 - (k1 .* k3) ./ (3 * k2 .^ 2);
        
        % Gaussian Transform as in [Mudholkar81]
        for ii = 1 : size(trainingSet, 1)
            T(ii, :) = (S2(ii, :) / k1(ii)) .^ h0(ii);
        end
        
        % when order1 is enforced, only the mean is expected to drift, the sample variance is thus only de-trended a priori and then
        % a constant is fitted
        if (strcmpi(params.featuresToCompute{ff}, 'gaussianizedSampleVariance')) || (strcmpi(params.featuresToCompute{ff}, 'gaussianizedSampleVarianceOrder1'))
            mu_hat = mean(T, 2);
            sigmaEst =  std(T')';
        end
        
        if (strcmpi(params.featuresToCompute{ff}, 'gaussianizedSampleVarianceLPA'))
            [mu_hat, sigmaEst] = zeroOrderFitParabolicWeights(T, sigma_noise);
        end
        
        % compute the second moment, the deviazione standard
        Theta(ff).expect = mu_hat;
        Theta(ff).stdDev = sigmaEst;
        
       
        if size(trainingSet, 1) == 2
            Theta(ff).transform = @(x1, x2) [(x1 ./ k1(1)) .^ h0(1) ; (x2 ./ k1(2) ) .^ h0(2)];
        end
        
        if size(trainingSet, 1) == 1
            Theta(ff).transform = @(x)((x ./ k1) .^ h0);
        end
        
        Theta(ff).h0 = h0;
        Theta(ff).k1 = k1;
        
    end
    
    Theta(ff).Gamma = Gamma;
    Theta(ff).featureName = params.featuresToCompute{ff};
end

end
