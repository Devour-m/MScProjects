
function [detected , testDetectingChange , Theta0 , dimensionWhereChangeDetected] =f_ICI_test_MultiDim(DataSet , tsLength, testParams, Gamma, Theta0)
%
% function [detected , testDetectingChange , Theta0 , dimensionWhereChangeDetected] = f_ICI_test_MultiDim(DataSet , tsLength , testParams , Gamma, Theta0)
%
% Component-wise change detection test based on ICI rule.
%
% input description
%
% DataSet                      - dataset vector to be analyzed. For Multidimensional data, each row of the dataset
%                                          should correspond to a component, thus
%                                          size(DataSet , 1) = sample dimension, (d in [1])
%                                          size(DataSet , 2) = number of samples of the dataset
% tsLength                     - length of training set. The training set is taken DataSet(: , 1 : tsLength)  (T_0 in [1])
% testParams                  - parameter structure provided by defineStandardParameters
%                                         windowWidth              - subsequences dimension (\nu in [1])
%                                         Gamma                        - Gamma , ICI parameter (in [1] as well)
%                                         weights                        - The weights used in the polynomial estimator ('uniform' [default], 'parabolic' or 'constant+alpha');
%                                         do_adaptiveGamma    - set this to 0 (optional, default 0)
%                                         do_show                      - plots results (optional, default 0)
%                                         do_debug_duro           - shows iteratively polynomial estimates, confidence intervals and their intersection (optional, default 0)
%                                         numberOfFeatures      - determines the number of features to be monitored by the
%                                                                               CDT (at the moment can be 0 or 1) 
%
% Gamma                        - Gamma parameter, when not used, Gamma = testParams.Gamma. Typically is used to specify a
%                                           different value of Gamma, e.g. for refinement
%
% numberOfFeatures     - when not speficied, numberOfFeatures = testParams.numberOfFeatures. Typically is used to specify a
%                                           different value of numberOfFeatures, e.g. for refinement
%
% Giacomo Boracchi
% Politecnico di Milano
% January 2010
% giacomo.boracchi@polimi.it
%
% please cite this work as
% [1] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
% "Change Detection Tests Using the ICI rule"
% in Proceedings of IJCNN 2010, the International Joint Conference on Neural Networks 18 - 23 July, 2010 Barcelona, Spain.
%
% [2] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
% "Adaptive Classifiers with ICI-based Adaptive Knowledge Base Management"
% ICANN 2010, 20th International Conference on Artificial Neural Networks, September 15-18, 2010, Thessaloniki, Greece,
% Lecture Notes on Computer Science. Springer Berlin / Heidelberg, vol 6353, pg 458-467
%
% [3] Cesare Alippi, Giacomo Boracchi and Manuel Roveri,
% "A just-in-time adaptive classification system based on the intersection of confidence intervals rule"
%  Neural Networks, Elsevier vol. 24 (2011), pp. 791-800 (doi: 10.1016/j.neunet.2011.05.012) 
%
% Revision History
% December 2010 -   changed input: passing test parameter structure
% December 2011 -   test can operate on a reduced number of features, depending on testParas.numberOfFeatures
% December 2013 -  Engineering
% December 2013 - added as an optional parameter Theta0, if passed as an argument computeTrainingFeatures is not invoked
% April 2014         - removed nubmerOfFeatures and weights, features are now specified in testParams.featuersToCompute

if (~exist('Gamma' , 'var') || isempty(Gamma))
    Gamma = testParams.Gamma;
end

if (~exist('Theta0' , 'var') || isempty(Theta0))
    do_computeTrainingFeatures = 1;
else
    do_computeTrainingFeatures = 0;
end

windowWidth = testParams.windowWidth;
do_show = testParams.do_show;
do_debug_duro = testParams.do_debug_duro;

testResult = zeros(size(DataSet));

% drawing parameters
LINE_WIDTH_BOLD = 3;
LINE_WIDTH_NORMAL = 2;

TEST_TO_PLOT1 = 1;
DIMENSION_TO_PLOT = 1;

figureNumber1 = 128;
figureNumber2 = 129;

do_compute_Values = 0;
do_show_iteration = 0;

intersection_interval_UP = [];
intersection_interval_DWN = [];

%% ICI based change detection : TS analysis and parameters estimation
% Compute local averages
% dataM = mean(reshape(DataSet,windowWidth,length(DataSet)/windowWidth))';
% trainData = dataM(1 : tsLength / windowWidth);



trainData = DataSet( : , 1 : tsLength);


% all the training features are computed in Theta0
if do_computeTrainingFeatures
    Theta0 = computeTrainFeatures_MultiDim(trainData, testParams, Gamma);
    % numel(Theta0) = 2 even for multidim 
end

% consider only the features to monitor
numberOfFeatures = numel(Theta0);

tsElements = floor(size(trainData , 2) / windowWidth);

% la struttura di intersection_inverval_UP e _DWN ?sulle righe cambio
% dimensione, sulle colonne cambio stima, sulla terza dimensione cambio il
% test (media varianza)
% size (intersection_interval_UP)  =  sampleDimension x #stimeCampioni x numberOfTest
% being sampleDimension = size(DataSet , 1);

for tt = 1 : numberOfFeatures %  numberOfFeatures is k in [1]
    intersection_interval_UP(: , 1 : tsElements , tt) = repmat((Theta0(tt).expect + Theta0(tt).Gamma * Theta0(tt).stdDev ) , [1 , tsElements , 1]);
    intersection_interval_DWN(: , 1 : tsElements , tt) =  repmat((Theta0(tt).expect - Theta0(tt).Gamma * Theta0(tt).stdDev ) , [1 ,  tsElements  , 1]);
end

% initialize confidence intervals - just for visualization purposes-
interval_UP = intersection_interval_UP;
interval_DWN = intersection_interval_DWN;

% data=reshape(DataSet,[size(DataSet,1),length(DataSet)/windowWidth,windowWidth]);
% prepare the data. Set the observation windows of \nu observations along the third dimension of data

for ii = 1 : size(DataSet,2) / windowWidth
    data(: , ii, :) = DataSet(: , (ii - 1) * windowWidth + 1 : (ii) * windowWidth);
end



% size (ExpectationVector) = sampleDimension x #stimeCampioni x numberOfTest
% size (StdVector) = sampleDimension x #stimeCampioni x numberOfTest
% being sampleDimension = size(DataSet , 1);

for tt = 1 : numberOfFeatures
    ExpectationVector( : , 1 : tsElements , tt) = repmat( Theta0(tt).expect , [1 , tsElements , 1]);
    StdVector( : , 1 : tsElements , tt ) = repmat(Theta0(tt).stdDev, [1 , tsElements , 1]);
end

%% ICI based change detection : operation phase

% the dataset contains only the TS 
if  size(data , 2) <=  tsElements
    dimensionWhereChangeDetected = 0;
    detected = 0;
    testDetectingChange = 0;
    %disp([''])
    %warning('Sollevo un falso negativo perch?il dataset corrisponde al ts');
end



for vs_len = tsElements + 1 : size(data , 2)
    
    % calcola le features correnti
    [ExpectationVector(: , vs_len , :) ,  StdVector(: , vs_len , :) , Theta1] = computeFeatures_MultiDim(data , ExpectationVector , Theta0 , testParams);
   
    % updates the coefficients
    if any(strcmpi(testParams.featuresToCompute, {'order1'}))
        Theta0 = Theta1;
    end
    
    % la deviazione standard ?sigma_0/sqrt(n_of_considered_samples)
    for tt = 1 : numberOfFeatures
        % compute the confidence interval
        interval_UP(: , vs_len , tt) = ExpectationVector(: , vs_len , tt) + Theta0(tt).Gamma *  StdVector(: , vs_len , tt);
        interval_DWN(: , vs_len , tt) = ExpectationVector(: , vs_len , tt) - Theta0(tt).Gamma *  StdVector(: , vs_len , tt);
    end
   
    
    % compute the intersection of confidenec interval
    [res , testDetectingChange , dimensionWhereChangeDetected] = checkIntersection_MultiDim( intersection_interval_UP(:,end,:), intersection_interval_DWN(:,end,:), interval_UP(:,end,:), interval_DWN(:,end,:));
    % ,intersection_interval_DWN(:,end,:), interval_UP(:,end,:), interval_DWN(:,end,:)
    % dimensionWhereChangeDetected
    
    % res==1 if there is still intersection among the intervals
    if(res)
        [intersection_interval_UP(: , vs_len , :) , intersection_interval_DWN(: , vs_len , :)] = computeIntesection_MultiDim(intersection_interval_UP(:,end,:), intersection_interval_DWN(:,end,:), interval_UP(:,end,:), interval_DWN(:,end,:));
        
        if do_debug_duro
            
            if do_compute_Values
                % sampleMeans
                dataM = mean(data , 2);
                % sampleVariance
                S2  =  mean((data - repmat(dataM , size(data , 1) , 1)) .^ 2);
                % compute the Gaussian approximating function
                dataT =  Theta0(2).transform(S2);
            else
                dataM = nan;
                dataT = nan;
            end
            
            if (TEST_TO_PLOT1 == 1)
                titleStr = 'Sample Mean';
            else
                titleStr = 'Sample Variance';
            end
            
            displayIndexes(ExpectationVector , StdVector , dataM , intersection_interval_UP , intersection_interval_DWN , interval_UP , interval_DWN , DIMENSION_TO_PLOT , TEST_TO_PLOT1 , figureNumber1 , LINE_WIDTH_BOLD , LINE_WIDTH_NORMAL , titleStr);
            %             displayIndexes(ExpectationVector , StdVector , dataM , intersection_interval_UP , intersection_interval_DWN , interval_UP , interval_DWN , DIMENSION_TO_PLOT , 1 , figureNumber1 + 1, LINE_WIDTH_BOLD , LINE_WIDTH_NORMAL , 'Sample Mean');
            
        end
        
    else
        % detected
        testResult(dimensionWhereChangeDetected , windowWidth * (vs_len - 1) : windowWidth * vs_len) = 1;
        
        break;
    end
    
end





%%
if do_show
    if do_compute_Values
        % sampleMeans
        dataM = mean(data);
        % sampleVariance
        S2  =  mean((data-repmat(dataM,size(data,1),1)).^2);
        % compute the Gaussian approximating function
        dataT =  Theta0(2).transform(S2);
    else
        dataM = nan;
        dataT = nan;
    end
    
    if (testDetectingChange == 1)
        titleStr = 'Sample Mean';
    else
        titleStr = 'Sample Variance';
    end
    
    % potrebbe essere che il dataset ?arrivato alla fine senza beccare un cambiamento, i.e. FP.
    if testDetectingChange > 0
        if length(testDetectingChange) > 1
            displayIndexes(ExpectationVector , StdVector , dataM , intersection_interval_UP , intersection_interval_DWN , interval_UP , interval_DWN , dimensionWhereChangeDetected(1) , testDetectingChange(1) , figureNumber1 , LINE_WIDTH_BOLD , LINE_WIDTH_NORMAL , 'Sample Mean');
            displayIndexes(ExpectationVector , StdVector , dataM , intersection_interval_UP , intersection_interval_DWN , interval_UP , interval_DWN , dimensionWhereChangeDetected(2) , testDetectingChange(2) , figureNumber2, LINE_WIDTH_BOLD , LINE_WIDTH_NORMAL , 'Sample Variance');
        else
            displayIndexes(ExpectationVector , StdVector , dataM , intersection_interval_UP , intersection_interval_DWN , interval_UP , interval_DWN , dimensionWhereChangeDetected(1) , testDetectingChange(1) , figureNumber1 , LINE_WIDTH_BOLD , LINE_WIDTH_NORMAL , titleStr);
        end
    else
        disp(['No change detected TS size: ' , num2str(tsElements)  , ', DataSet size: ' , num2str(size(data , 2))]);
    end
end

% potrebbe esserci un problema quando viene beccato il cambiamento
% contemporaneamente nello stesso slot per componenti diversi delle
% osservazioni. In tal caso si impone che la dimensione che becca il
% cambiamneto sia la prima
dimensionWhereChangeDetected = dimensionWhereChangeDetected(1);
%(nel caso in cui non avvenissero detection simultanee, il problema non si pone)

%idem per testDetectingChange
testDetectingChange = testDetectingChange(1);


% outputValues
if dimensionWhereChangeDetected == 0
    detected = 0;
else
    detected = find(testResult(dimensionWhereChangeDetected,:)>0 , 1 , 'last');
end

% questo gi?contiene l'identificativo del test che non ha passato il test
if numel(testDetectingChange>1) && isempty(detected)
    testDetectingChange = sum(testDetectingChange);
end



%% plot results
% Dataset
if do_show_iteration
    N = size(DataSet,2);
    
    if detected
        if(detected >= N * changePoint(dimensionWhereChangeDetected))
            disp(['ICI-test: Change Detected at ',num2str(detected)]);
        else
            disp(['ICI-test: False Positive: Change Detected at ',num2str(detected)]);
        end
    else
        detected = 0;
        disp('ICI-test: False Negative: Change Missed. ');
    end
end

