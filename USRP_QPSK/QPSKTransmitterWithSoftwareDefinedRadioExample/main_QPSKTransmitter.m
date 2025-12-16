sdrType = "USRP";
SDRName = "N320/N321";
SDRAddress = '192.168.10.2';
sampleRate         = 1000000;  % Sample rate of transmitted signal
SDRGain            = 40;  % Set radio gain
SDRCenterFrequency = 915000000;  % Set radio center frequency
SDRStopTime        = 10;  % Radio transmit time in seconds

availableRadios = helperFindRadios(SDRAddress);

% Transmitter parameter structure
prmQPSKTransmitter = sdrQPSKTransmitterInit(SDRName, SDRAddress, sampleRate, SDRCenterFrequency, ...
    SDRGain, SDRStopTime);

underruns = runSDRQPSKTransmitter(prmQPSKTransmitter);

fprintf('Total number of underruns = %d.\n', underruns);
