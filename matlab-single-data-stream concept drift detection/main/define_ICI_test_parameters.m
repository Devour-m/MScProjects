function params = define_ICI_test_parameters(tsLength, featuresToCompute, windowWidth, HTname)
% function params  = define_ICI_test_parameters(tsLength, featuresToCompute, windowWidth, HTname)
%
%
% --- Input Description
% tsLength            - number of training samples provided (initial part of the training set)
%
% featuresToCompute   - cell array containing the identifiers of the features to be comptued 
%                       these are
%                           'sampleMeans', 'BoxCox', 'Manly', 'polyFitOrder1', 
%                           'sampleMeansLPA', (TO DETECT CHANGES IN THE PROCESS EXPECTATION/LOCATION)
%                           'classificationError', 'gaussianizedSampleVariance', 'gaussianizedSampleVarianceOrder1', 
%                           'gaussianizedSampleVarianceLPA', (TO DETECT CHANGES IN THE PROCESS VARIANCE/SCALE)
%                       [optional] default value is set as in [1]
%                           params.featuresToCompute{1} = 'sampleMeans';
%                           params.featuresToCompute{2} = 'gaussianizedSampleVariance';
%                      More about the features can be found in the refernces, in particular
%                           'sampleMeans', 'sampleMeansLPA', 'gaussianizedSampleVariance', 'gaussianizedSampleVarianceLPA', are presented in [1]
%                           'polyFitOrder1', 'gaussianizedSampleVarianceOrder1'  are discussed in [3]
%                           'BoxCox', 'Manly', are discussed in [5]
%                           'classificationError' is discussed in [2]
%
% windowWidth           - size of the (running) window to be used for feature extraction. It can be set also to 1 for
%                           pointwise procesisng when  'BoxCox', 'Manly', features are used, as discussed in [5]
%                           [optional] default value is set to 20 as in [1]
%
% HTname                - name of the Hypothesis Test used at the validation layer
%                           HT_Name = 'ztest'; % ztest on the feature mean
%                           HT_Name = 'Hotelling'; % multivariate on Gaussian-distributed features
%                           HT_Name = 'Lepage'; % a CPM runs independently on each feature to detect changes in location an
%                                       scale (see Ross 2011, Technometrics and Lepage 1974)
%                           HT_Name = 'Proportions'; % univariate, meant for Bernoulli distributed observations
%
% ---- Output Description
% params                - structure containing all the parameters for ICI-based CDT configuration. See source code for a detailed description
%
% Giacomo Boracchi
% Politecnico di Milano
% January 2010 
% giacomo.boracchi@polimi.it
%
% References
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
% [4] Cesare Alippi, Giacomo Boracchi, Manuel Roveri,
% Hierarchical Change-Detection Tests 
% IEEE Transactions on Neural Networks and Learning Systems, In Press (2016), 13 pages doi:10.1109/TNNLS.2015.2512714 
%
% [5] Giacomo Boracchi, Manuel Roveri
% A Reconfigurable and Element-wise ICI-based Change-Detection Test for Streaming Data
% CIVEMSA 2014, Computational Intelligence and Virtual Environments for Measurements Systems and Applications
%
%
% Revision History
% January 2011 - taken from defineStandardParameters()
% January 2011 - Added alpha for hypothesist test paramseters.alpha (parameters.alpha previousy refers to cicusum)
% November 2013 - Enhancement, added some controls, removed dataset length from the input, added weights
% April 2014 -  Introduced the field featuresToCompute to selectively determine which features to compute, removed
% numberofFeatures

params.tsLength = tsLength;

if ~exist('featuresToCompute', 'var') || isempty(featuresToCompute)
    featuresToCompute{1} = 'sampleMeans';
    featuresToCompute{2} = 'gaussianizedSampleVariance';
end

% in case a single string is provided as input
if ~iscell(featuresToCompute)
    featuresToComputeCell{1} = featuresToCompute;
    featuresToCompute = featuresToComputeCell;
end

if ~exist('windowWidth', 'var') || isempty(windowWidth)
    windowWidth = 20;
end

if ~exist('HTname', 'var') || isempty(HTname)
    HTname = 'Lepage';
end


% here are the features implemented
% 'sampleMeans', 'BoxCox', 'Manly', 'polyFitOrder1', 'sampleMeansLPA',
% 'classificationError',
% 'gaussianizedSampleVariance', 'gaussianizedSampleVarianceOrder1', 'gaussianizedSampleVarianceLPA',
params.featuresToCompute = featuresToCompute;
params.windowWidth = windowWidth;

% ICI-based CDT parameters on features extracted from raw data
params.do_adaptiveGamma = 0;
params.Gamma = 2; %2

% ICI-based CDT parameters for the REFINEMENT PROCEDURE on features extracted from raw data
params.GammaRefinement = 1.75; % in [4] we set GammaRefinement = Gamma - 0.25
params.GammaReverseAnalysis = 2;

% ICI-based CDT parameters on the classfication error
params.Gamma_err = 2;
params.GammaRefinement_err = 2;
params.GammaReverseAnalysis_err = 2;

params.K_refinement = 1.5;
params.MinimumTS_Size = 80;
%params.MinimumTS_Size_CB = 80;
% params.MinimumTS_Size_err = 80;
params.MinimumTS_Size = params.windowWidth * round(params.MinimumTS_Size / params.windowWidth);

% parameters used only for training the classifier K_0, the classifier used to determine the classification errors
params.CV_ratio = 2; % used only as parameter to train the classifier K_0 to monitor stationarity in the classifier error
params.do_retrain_K0 = 1; % used to determine if the classifier K_0 has to be trainied after every detecion on the new trainig set

% Hypothesis test parameters for the hierarchical procedure
params.alpha = 0.05;

% hypothesis test used to validate cdt results
params.HT_Name = HTname; % Analyzes the raw scalar observations and runs a Lepage CPM

if (strcmpi(params.HT_Name, 'Lepage'))
    params.cpmParams = define_CPM_parameters(params.HT_Name);
    params.alpha = params.cpmParams.ALPHA;
end

% criteria to partition the dataset according to t_ref
params.HT_Partition = 'TrainingSet'; % considers only the training set as generated in stationary conditions
% e.g. oldTS = [1 , TS_length] ;  newTS = [T_ref , T_hat]
% parameters.HT_Partition = 'DataSet'; % considers only all the samples before T_ref as generated in stationary conditions
% e.g. oldTS = [1 , T_ref] ;  newTS = [T_ref + 1 , T_hat]

%% PointWise CDT: Gassian Approximativel Tranform
% params.pointwise.transformType = 'manly'; % 'boxcox', 'manly' (requires R)
params.pointwise.BoxCoxOffset = 0.5;


%% classifier-related parameters
params.classifierToUse = 'knn';%'svm' or 'knn','nb' , classifier to be used as K_0, just for concept-drift detection purposes
params.classifierToUseForComparingConcepts = 'knn';%'lda' or 'knn' or 'svm' or'nb'
params.retrain_K0_mode = 'yes'; % retrain K_0 after each detection

%parameters.errorComparisonMode =  'partitioning';%'sogliabrutale'%'crossequivalence';%'equality';%'partitioning'
%parameters.minimumConceptsSamples = 1000;
%parameters.minimumConceptsSupervisedSamples = 200;

% parameters for handling recurrent concepts
params.errorComparisonMode =  'crossequivalence';%'sogliabrutale'%'crossequivalence';%'equality';%'partitioning'
params.minimumConceptsSamples = 10;
params.minimumConceptsSupervisedSamples = 10;

params.classifier_parameters.k_estimate_mode = 'sqrt'; %sqrt fixed loo
params.classifier_parameters.svmTrainingMethod = 'SMO';
params.classifier_parameters.alpha = 0.25;
params.classifier_parameters.SOGLIA_BRUTALE = 0.3;

% TOST parameters
params.classifier_parameters.alpha_equality = 0.1; % alpha del TOST per i dati e l'errore
params.classifier_parameters.delta_equality_data = 0.3; % scaling factor w.r.t the reference mean
params.classifier_parameters.delta_equality_err = 0.4; % scaling factor w.r.t the reference mean (used only when comparionsMode == cr)

% crossequivalence parameters
params.classifier_parameters.p_val_threshold_err = 0.2;
params.classifier_parameters.p_val_only = 0;
params.classifier_parameters.perc_only = 1;
params.classifier_parameters.p_val_plus_perc = 0;
params.classifier_parameters.p_val_and_perc = 0;
params.classifier_parameters.perc_equality_threshold = 0.8;
params.classifier_parameters.PartitioningRatioForCrossEquivalence = 2;
params.classifier_parameters.compareWithoutPartitioning = 1;
params.classifier_parameters.minSamplesPerLabel = 2;

% both crossequivalence and partitioning
params.classifier_parameters.MergeKBForConceptComparison = 0;

% partitioning parameters
params.classifier_parameters.numberKFoldValidation = 50;
params.classifier_parameters.CumulativePercentagePartitioning = 0.8; %0.85 % il valore del threshold ?il
params.classifier_parameters.PartitioningRatioForPartitioning = 3;
params.classifier_parameters.MaximumPercentageMissedCorrespondences = 0.25; %0.2

%% procedural parameters
params.do_show = 0;
params.do_save = 0;
params.do_disp = 0;
params.do_debug_duro = 0;

params.do_test_class_error = 1; % uses CDT_eps
params.do_test_data = 0; % uses CDT_x
params.do_stop_first_detection = 0; % if do_validate_changes  == 1 stops at the first validated detections, otherwise at the first detection
params.do_validate_changes = 1; % enable validation layer (typically not used in classification applications)

%% cross check

% use only univariate tests on point-wise methods
if any(strcmp('BoxCox', params.featuresToCompute)) || any(strcmp('Manly', params.featuresToCompute))
    if strcmpi(params.HT_Name, 'ztest') == 0 && strcmpi(params.HT_Name, 'Lepage') == 0 && strcmpi(params.HT_Name, 'Mood') == 0 && strcmpi(params.HT_Name, 'Mann-Withney') == 0
        params.HT_Name = 'ztest';
    end
end

% on point-wise methods, only windowWidth = 1 are allowed
if (any(strcmp('BoxCox', params.featuresToCompute)) || any(strcmp('Manly', params.featuresToCompute))) && params.windowWidth > 1
    warning('typically params.windowWidth  = 1 is expected when using point-wise features')
end

% sample variance cannot be computed on pointwise features
if (any(strcmp(params.featuresToCompute, 'gaussianizedSampleVariance')) || ...
        any(strcmp(params.featuresToCompute, 'gaussianizedSampleVarianceOrder1')) || ...
        any(strcmp(params.featuresToCompute, 'gaussianizedSampleVarianceLPA'))) && params.windowWidth < 3 
        error([' Sample variance cannot be computed when params.windowWidth = ', num2str(params.windowWidth)])
end

% when monitoring the classification error use only the classification error
if any(strcmp('classificationError', params.featuresToCompute)) && numel(params.featuresToCompute) > 1
    params.featuresToCompute{1} = 'classificationError';
    params.featuresToCompute(2 : end) = [];
    disp('using only one feature on Bernoulli observation')
end

% when monitoring the classification error use HT on proportions for validation purposes
if any(strcmp('classificationError', params.featuresToCompute)) && strcmpi(params.HT_Name, 'proportions') == 0
    params.HT_Name = 'Proportions';
    disp('change validation using HT on proportions')
end

% output messages
if params.do_stop_first_detection && params.do_validate_changes
    disp(['cdt stops at the first VALIDATED detection; do_stop_first_detection = ', num2str(params.do_stop_first_detection), ' do_validate_changes = ', num2str(params.do_validate_changes)'])
elseif params.do_stop_first_detection
    disp(['cdt stops at the first detection; do_stop_first_detection = ', num2str(params.do_stop_first_detection), ' do_validate_changes = ', num2str(params.do_validate_changes)'])
end


