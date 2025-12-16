clear all
sdrType = "USRP";
SDRName = "N320/N321";
SDRAddress = '192.168.10.2';
sampleRate         = 1000000;  % Sample rate of transmitted signal
SDRGain            = 30;  % Set radio gain
SDRCenterFrequency = 915000000;  % Set radio center frequency
SDRStopTime        = 10;  % Radio transmit time in seconds
previewReceivedData = true;  % enable to preview the received data 
printReceivedData   = true;  % enable to print the received data
isHDLCompatible     = false;  % disable to run 'FFT-Based' coarse frequency compensation 
% instead of 'Correlation-Based' for improved performance in MATLAB version.

availableRadios = helperFindRadios(SDRAddress);

% Transmitter parameter structure
prmQPSKTransmitter = sdrQPSKTransmitterInit(SDRName, SDRAddress, sampleRate, SDRCenterFrequency, ...
    SDRGain, SDRStopTime);

prmQPSKReceiver = sdrQPSKReceiverInit(SDRName, SDRAddress, sampleRate, SDRCenterFrequency, ...
    SDRGain, SDRStopTime, isHDLCompatible);

[underruns,BER, overflow, output] = runSDRQPSK_text(prmQPSKTransmitter,prmQPSKReceiver, previewReceivedData, printReceivedData);

fprintf('Total number of underruns = %d.\n', underruns);

fprintf('Error rate is = %f.\n', BER(1));
fprintf('Number of detected errors = %d.\n', BER(2));
fprintf('Total number of compared samples = %d.\n', BER(3));
fprintf('Total number of overflows = %d.\n', overflow);
