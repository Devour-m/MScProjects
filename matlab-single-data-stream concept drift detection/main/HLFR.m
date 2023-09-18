function [detect_time_permutation] = HLFR(InputData)
    % Hierarchical LFR with relearned SVM: single-stream concpet drift detection algorithm based on the false alarm rate
    
    addpath('../concept drift detection toolbox')

    % HLFR hyperparameters
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

end
    