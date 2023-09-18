clear all
clc
addpath('../../../linspecer')  
addpath('../libsvm-3.23')  
addpath('../../../concept drift detection toolbox')  
addpath(genpath(pwd))

%% data generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [num,txt,raw]= xlsread('usenet1.xlsx');
% data = num(2:end,2:end);
% change_points = [300, 600, 900, 1200]; % locations of change points
% % 
% [num,txt,raw]= xlsread('hyperplane.xlsx');
% data = num(1:59999,2:end);
% change_points = [10000, 20000, 30000, 40000, 50000]; % locations of change points
% [data, change_points] = data_selection(data, change_points, 1000);

% [num,txt,raw]= xlsread('sea.xlsx');
% data = num;
% change_points = [15000, 30000, 45000]; % locations of change points
% [data, change_points] = data_selection(data, change_points, 1000);

N = 1000; % number of samples in each concept
 data = ConceptDriftData('abrupt',11,N,0.5);
 change_points = [N,2*N,3*N,4*N,5*N];

N = 5000; % number of samples in each concept
% data = ConceptDriftData_new('checkerboard',N);
change_points = [N,2*N,3*N,4*N,5*N];


plot(1:16000,data{1:16000,2})
autocorr(data{8000:10000,17},100)
autocorr(data{1:500000,6},100)

autocorr(data{1:40000,2},20000)

datp = 'airlines';
type = [];
sev = [];
i=1;
[from_dir, datanm] = ReadInputData(datp, sev, type,1);
flnm = [from_dir, datanm];
data=readtable(flnm);

%% load from file %%
flnm = [from_dir, datanm];
if exist(flnm, 'file')
    data = load(flnm);
else
    error(['Error: ', flnm, ' not included.']);
end





% generate training and testing dataset
% generate training samples
index = randperm(change_points(1));
num_least_relearn = 160;
distance_threshold = 30;
train_index = index(1:num_least_relearn);
train_data = data(train_index,:);

ini_num_least_relearn = 160;
train_data = data(1:ini_num_least_relearn,:);
test_data = data((ini_num_least_relearn+1):end,:);


% generate testing samples
 test_data = data([index(num_least_relearn+1:end),change_points(1)+1:end],:);
warn_coeff = 2;
detect_coeff = 3;
 [detect_time_DDM,classification_results_DDM,error_rate] = ...
    DDM_detection(train_data,test_data,warn_coeff,...
    detect_coeff,num_least_relearn,'real',distance_threshold);

  [detect_time_DDM,classification_results_DDM,error_rate] = ...
    CBHCDT_f_DDM_detection(train_data,test_data,warn_coeff,...
    detect_coeff,num_least_relearn,'real',distance_threshold,'SVM_linear');

 
 
 
%% online learning with different methods
% Hierarchical LFR with relearned SVM
eta = [0.9,0.9,0.9,0.9];
permutation_window = 200;
tic;
[detect_time_permutation,classification_results_permutation,...
    tpr_permutation,tnr_permutation,ppv_permutation,npv_permutation] = ...
    LFR_permutation_detection(train_data,test_data,eta,...
    num_least_relearn,'real',4,distance_threshold,permutation_window);
toc;

[fitdetect_time_permutation,fitclassification_results_permutation,...
    fittpr_permutation,fittnr_permutation,fitppv_permutation,fitnpv_permutation] = ...
    CB_HCDT_f_LFR_permutation_detection(train_data,test_data,eta,...
    num_least_relearn,'real',4,distance_threshold,permutation_window,'SVM_RBF');

tic;
% LFR with relearned SVM
[detect_time_relearn,classification_results_relearn,...
    tpr_relearn,tnr_relearn,ppv_relearn,npv_relearn] = ...
    LFR_detection(train_data,test_data,eta,...
    num_least_relearn,'real',4,distance_threshold);
toc;

% DDM
tic;
warn_coeff = 2;
detect_coeff = 3;
[detect_time_DDM,classification_results_DDM,error_rate] = ...
    DDM_detection(train_data,test_data,warn_coeff,...
    detect_coeff,num_least_relearn,'real',distance_threshold);
toc;

[fitdetect_time_DDM,fitclassification_results_DDM,fiterror_rate] = ...
    CBHCDT_f_DDM_detection(train_data,test_data,warn_coeff,...
    detect_coeff,num_least_relearn,'real',distance_threshold,'SVM_rbf');


% EDDM
tic;
warn_coeff_EDDM = 0.95;
detect_coeff_EDDM = 0.90;
min_num_errors = 30;
[detect_time_EDDM,classification_results_EDDM,distance_rate] = ...
    EDDM_detection(train_data,test_data,warn_coeff_EDDM,...
    detect_coeff_EDDM,num_least_relearn,'real',min_num_errors,distance_threshold);
toc;

[detect_time_EDDM,classification_results_EDDM,distance_rate] = ...
    CBHCDT_f_EDDM_detection(train_data,test_data,warn_coeff_EDDM,...
    detect_coeff_EDDM,num_least_relearn,'real',min_num_errors,distance_threshold,'SVM_linear');


% STEPD
tic;
warn_coeff_STEPD = 0.05;
detect_coeff_STEPD = 0.003;
window_size = 30;
[detect_time_STEPD,classification_results_STEPD,error_rate] = ...
    STEPD_detection(train_data,test_data,warn_coeff_STEPD,...
    detect_coeff_STEPD,num_least_relearn,'real',distance_threshold,window_size);
toc;

% DDM_OCI
tic;
warn_coeff_OCI = 5; % default 10 20
detect_coeff_OCI = 10; % default 15 30
eta_OCI = 0.9;
[detect_time_DDM_OCI,classification_results_DDM_OCI,error_rate] = ...
    DDM_OCI_detection(train_data,test_data,warn_coeff_OCI,...
    detect_coeff_OCI,eta_OCI,num_least_relearn,'real',distance_threshold);
toc;

%% analyze results
% plot detected time index
detect_results_permutation.values = detect_time_permutation;
detect_results_permutation.method = 'Hierarchical LFR';
detect_results_relearn.values = detect_time_relearn;
detect_results_relearn.method = 'LFR';
detect_results_DDM.values = detect_time_DDM;
detect_results_DDM.method = 'DDM';
detect_results_EDDM.values = detect_time_EDDM;
detect_results_EDDM.method = 'EDDM';
detect_results_STEPD.values = detect_time_STEPD;
detect_results_STEPD.method = 'STEPD';
detect_results_DDM_OCI.values = detect_time_DDM_OCI;
detect_results_DDM_OCI.method = 'DDM-OCI';
change_results.values = change_points;
change_results.method = 'Groud truth';
detect_time = {detect_results_permutation,detect_results_relearn,...
    detect_results_DDM,detect_results_EDDM,detect_results_STEPD,...
    detect_results_DDM_OCI,change_results};
test_L = length(test_data);
show_results(detect_time,test_L,num_least_relearn);