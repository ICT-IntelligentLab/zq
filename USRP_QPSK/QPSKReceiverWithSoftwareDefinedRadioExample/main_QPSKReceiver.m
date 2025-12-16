sdrType = "USRP";
SDRName = "N320/N321";
SDRAddress = '192.168.10.2';
sampleRate          = 1000000;
SDRGain             = 30;
SDRCenterFrequency  = 915000000;
SDRStopTime         = 10;
previewReceivedData = true;  % enable to preview the received data 
printReceivedData   = true;  % enable to print the received data
isHDLCompatible     = false;  % disable to run 'FFT-Based' coarse frequency compensation 
% ins
% 
% 
% tead of 'Correlation-Based' for improved performance in MATLAB version.

availableRadios = helperFindRadios(SDRAddress);

prmQPSKReceiver = sdrQPSKReceiverInit(SDRName, SDRAddress, sampleRate, SDRCenterFrequency, ...
    SDRGain, SDRStopTime, isHDLCompatible);

[BER, overflow, output] = runSDRQPSKReceiver(prmQPSKReceiver, previewReceivedData, printReceivedData);

fprintf('Error rate is = %f.\n', BER(1));
fprintf('Number of detected errors = %d.\n', BER(2));
fprintf('Total number of compared samples = %d.\n', BER(3));
fprintf('Total number of overflows = %d.\n', overflow);