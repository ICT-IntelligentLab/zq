function connectedRadios = helperFindRadios(SDRAddress)
%   Copyright 2023-2025 The MathWorks, Inc.
fctrlCond = matlab.internal.feature("findsdru", 1); %#ok<NASGU>
connectedRadios = findsdru(SDRAddress);
fprintf("\n")
disp(connectedRadios);
end