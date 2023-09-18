function [HCDT_conDetections] = HCDT(InputData)
    % HCDT: hierarical concept drift detection algorithm that is based on data features
    % HCDT_conDetections: confirmed detections: when #? data (? starts from 1) instanace arrives,
    % the concept drift occurs in the single data stream
    
    InputData = InputData';

    % HCDT hyperparameters
    tsLength = 160;
    cdtParams = define_ICI_test_parameters(tsLength, [], [], 'Hotelling'); % tune parameters here
    cdtParams.MinimumTS_Size =160;
    cdtParams.GammaRefinement = 2.25; % higher Gamma = fewer detections
    cdtParams.Gamma = 2.5;
    PCAactive = 0;
    PCApercent = 70;

    tic;
    t1 = toc;
    [HCDT_conDetections, HCDT_oriDetections, HCDT_dimensionDetected, ...
        HCDT_tsEnd, HCDT_tsInit, HCDT_tsLengths, HCDT_testDetectiongChange] = ...
        f_ICI_test_MultiChange_MultiStage(InputData,  cdtParams.tsLength , cdtParams,PCAactive,PCApercent);
    t2 = toc;
    runtime_HCDT = t2-t1;

end
    