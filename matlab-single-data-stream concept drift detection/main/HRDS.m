function [HRDS_conDetections] = HRDS(InputData)
%HRDS shuyi's single-stream concept drift detection algorithm
%   HRDS_conDetections: confirmed detections: when #? data (? starts from 1) instanace arrives,
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

end

