%% demo for histogram comparison between LFR and LFR_new

clear
clc
addpath(genpath(pwd))

%% initialization
num_mc = 2; % number of monte-carlo simulations
num_least_relearn = 100;
distance_threshold = 50;

detect_time_total_permutation = []; % vector to record all the detected time of concept drift with H-LFR
detect_time_total_relearn = []; % vector to record all the detected time of concept drift with LFR
detect_time_total_DDM = []; % vector to record all the detected time of concept drift with DDM
detect_time_total_EDDM = []; % vector to record all the detected time of concept drift with EDDM
detect_time_total_STEPD = []; % vector to record all the detected time of concept drift with STEPD
detect_time_total_DDM_OCI = []; % vector to record all the detected time of concept drift with DDM OCI

%% data generation
% [num,txt,raw]= xlsread('usenet1.xlsx');
% data = num(2:end,2:end);
% change_points = [300, 600, 900, 1200]; % locations of change points

N = 1000; % number of samples in each concept
data = ConceptDriftData_new('checkerboard',N);
change_points = [N,2*N,3*N,4*N,5*N,6*N,7*N];

% [num,txt,raw]= xlsread('hyperplane.xlsx');
% data = num(1:end,2:end);
% change_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]; % locations of change points
% % % [data, change_points] = data_selection(data, change_points, 1000);

% [num,txt,raw]= xlsread('sea.xlsx');
% data = num;
% change_points = [15000, 30000, 45000]; % locations of change points
% [data, change_points] = data_selection(data, change_points, 1000);

% N = 1000; % number of samples in each concept
% data = ConceptDriftData('abrupt',11,N,0.5);
% change_points = [N,2*N,3*N,4*N,5*N];

%% Monte-Carlo simulations
t_HLFR = [];
t_LFR = [];
t_DDM = [];
t_EDDM = [];
t_STEPD = [];
t_DDM_OCI = [];

for i = 1:num_mc
    fprintf('Monte-Carlo simulation trail %1.0f\n',i);
% generate training and testing dataset
% generate training samples
index = randperm(change_points(1));

train_index = index(1:num_least_relearn);
train_data = data(train_index,:);

% generate testing samples
test_data = data([index(num_least_relearn+1:end),change_points(1)+1:end],:);

% Hierarchical LFR with relearned SVM
tic;
eta = [0.9,0.9,0.9,0.9];
permutation_window = 200;
[detect_time_permutation,classification_results_permutation,...
    tpr_permutation,tnr_permutation,ppv_permutation,npv_permutation] = ...
    LFR_permutation_detection(train_data,test_data,eta,...
    num_least_relearn,'real',4,distance_threshold,permutation_window);
t_end = toc;
t_HLFR = [t_HLFR, t_end/numel(detect_time_permutation)];

% LFR with relearned SVM
tic;
[detect_time_relearn,classification_results_relearn,...
    tpr_relearn,tnr_relearn,ppv_relearn,npv_relearn] = ...
    LFR_detection(train_data,test_data,eta,...
    num_least_relearn,'real',4,distance_threshold);
t_end = toc;
t_LFR = [t_LFR, t_end/numel(detect_time_relearn)];

% DDM
tic;
warn_coeff = 2;
detect_coeff = 3;
[detect_time_DDM,classification_results_DDM,error_rate] = ...
    DDM_detection(train_data,test_data,warn_coeff,...
    detect_coeff,num_least_relearn,'real',distance_threshold);
t_end = toc;
t_DDM = [t_DDM, t_end/numel(detect_time_DDM)];

% EDDM
tic;
warn_coeff_EDDM = 0.95;
detect_coeff_EDDM = 0.90;
min_num_errors = 60;
[detect_time_EDDM,classification_results_EDDM,distance_rate] = ...
    EDDM_detection(train_data,test_data,warn_coeff_EDDM,...
    detect_coeff_EDDM,num_least_relearn,'real',min_num_errors,distance_threshold);
t_end = toc;
t_EDDM = [t_EDDM, t_end/numel(detect_time_EDDM)];

% STEPD
tic;
warn_coeff_STEPD = 0.05;
detect_coeff_STEPD = 0.003;
window_size = 30;
[detect_time_STEPD,classification_results_STEPD,error_rate] = ...
    STEPD_detection(train_data,test_data,warn_coeff_STEPD,...
    detect_coeff_STEPD,num_least_relearn,'real',distance_threshold,window_size);
t_end = toc;
t_STEPD = [t_STEPD, t_end/numel(detect_time_STEPD)];

% DDM_OCI
tic;
warn_coeff_OCI = 5; % default 10
detect_coeff_OCI = 10; % default 15
eta_OCI = 0.9;
[detect_time_DDM_OCI,classification_results_DDM_OCI,error_rate] = ...
    DDM_OCI_detection(train_data,test_data,warn_coeff_OCI,...
    detect_coeff_OCI,eta_OCI,num_least_relearn,'real',distance_threshold);
t_end = toc;
t_DDM_OCI = [t_DDM_OCI, t_end/numel(detect_time_DDM_OCI)];

%% record all the concept drift detection time
detect_time_total_permutation = [detect_time_total_permutation detect_time_permutation];
detect_time_total_relearn = [detect_time_total_relearn detect_time_relearn];
detect_time_total_DDM = [detect_time_total_DDM detect_time_DDM];
detect_time_total_EDDM = [detect_time_total_EDDM detect_time_EDDM];
detect_time_total_STEPD = [detect_time_total_STEPD detect_time_STEPD];
detect_time_total_DDM_OCI = [detect_time_total_DDM_OCI detect_time_DDM_OCI];
end
%%

figure,
subplot(611),
hist(detect_time_total_permutation,10000);
xlim([0 80000]);
hold on
for j=1:length(change_points)
    line([change_points(j)-num_least_relearn change_points(j)-num_least_relearn],...
        ylim,'Color','r');
end
hold off

subplot(612)
hist(detect_time_total_relearn,100);
xlim([0 8000]);
hold on
for j=1:length(change_points)
    line([change_points(j)-num_least_relearn change_points(j)-num_least_relearn],...
        ylim,'Color','r');
end
hold off

subplot(613)
hist(detect_time_total_DDM,100);
xlim([0 8000]);
hold on
for j=1:length(change_points)
    line([change_points(j)-num_least_relearn change_points(j)-num_least_relearn],...
        ylim,'Color','r');
end
hold off

subplot(614)
hist(detect_time_total_EDDM,100);
xlim([0 8000]);
hold on
for j=1:length(change_points)
    line([change_points(j)-num_least_relearn change_points(j)-num_least_relearn],...
        ylim,'Color','r');
end
hold off

subplot(615)
hist(detect_time_total_STEPD,100);
xlim([0 8000]);
hold on
for j=1:length(change_points)
    line([change_points(j)-num_least_relearn change_points(j)-num_least_relearn],...
        ylim,'Color','r');
end
hold off

subplot(616)
hist(detect_time_total_DDM_OCI,100);
xlim([0 8000]);
hold on
for j=1:length(change_points)
    line([change_points(j)-num_least_relearn change_points(j)-num_least_relearn],...
        ylim,'Color','r');
end
hold off

mean(t_HLFR)
mean(t_LFR)
mean(t_DDM)
mean(t_EDDM)
mean(t_STEPD)
mean(t_DDM_OCI)