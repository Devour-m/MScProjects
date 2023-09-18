
% read inputdata
datanm = ['multi_drifts_6D_1.csv'];
InputData = load(datanm);


addpath('../concept drift detection toolbox')
% Hierarchical LFR with relearned SVM: single-stream concpet drift detection algorithm
% based on the false alarm rate

eta = [0.9,0.9,0.9,0.9];
permutation_window = 200;
num_least_relearn = 160; 
distance_threshold = 30;
train_data = InputData(1:num_least_relearn,:);
test_data = InputData((num_least_relearn+1):end,:); 
tic;
t1 = toc;
[detect_time_permutation,classification_results_permutation,...
    tpr_permutation,tnr_permutation,ppv_permutation,npv_permutation] = ...
    CBHCDT_f_LFR_permutation_detection(train_data,test_data,eta,...
    num_least_relearn,'real',4,distance_threshold,permutation_window,'SVM_linear');
t2 = toc;
runtime_permutation = t2-t1;

% HRDS: shuyi's single-stream concept drift detection algorithm
% HRDS_conDetections: confirmed detections: when #? data (? starts from 1) instanace arrives,
% the concept drift occurs in the single data stream

InputData = InputData';
K = 'linear';
nF = 1;
ts = 0.01;
tsLength = 160;
cdtParams = define_ICI_test_parameters(tsLength, [], [], 'Hotelling');
cdtParams.MinimumTS_Size =160;
cdtParams.GammaRefinement = 2.25; % higher Gamma = fewer detections
cdtParams.Gamma = 2.5;
tic;
t1 = toc;
[HRDS_conDetections, HRDS_oriDetections, HRDS_dimensionDetected, HRDS_ClassDetected, HRDS_tsEnd, HRDS_tsEnd0, HRDS_tsEnd1, ...
    HRDS_tsInit, HRDS_tsInit0, HRDS_tsInit1, HRDS_tsLengths, HRDS_tsLengths0, HRDS_tsLengths1, HRDS_testDetectingChange] =...
    CBHCDT_f_finalRCBM_scheme2(K, nF, ts, InputData,  tsLength , cdtParams);
t2 = toc;
runtime_HRDS = t2-t1;

% HCDT: single-stream concept drift detection algorithm based on data features

PCAactive = 0;
PCApercent = 70;
tic;
t1 = toc;
[HCDT_conDetections, HCDT_oriDetections, HCDT_dimensionDetected, ...
    HCDT_tsEnd, HCDT_tsInit, HCDT_tsLengths, HCDT_testDetectiongChange] = ...
    f_ICI_test_MultiChange_MultiStage(InputData(1:end-1,:),  cdtParams.tsLength , cdtParams, PCAactive, PCApercent);
t2 = toc;
res.HCDT_runtime = t2-t1;