clear; clc;

%% ====== 接收参数（必须与发射端完全一致）======
ipAddress       = '192.168.10.3';   % 这台电脑连接的 N321 IP（电脑 B 控制的接收设备）
centerFreq      = 2.4e9;            % 与发射端相同
gainRx          = 50;               % 接收增益（根据信号强度调整）
masterClockRate = 200e6;
sampleRate      = 10e6;
decimFactor     = masterClockRate / sampleRate;  % 20
frameLength     = 1e5;

%% ====== 创建接收器 ======
rx = comm.SDRuReceiver(...
    'Platform', 'N320/N321', ...
    'IPAddress', ipAddress, ...
    'CenterFrequency', centerFreq, ...
    'Gain', gainRx, ...
    'MasterClockRate', masterClockRate, ...
    'DecimationFactor', decimFactor, ...
    'ChannelMapping', 1, ...    % 推荐用 2（RX2 端口）以获得更好隔离
    'OutputDataType', 'double', ...
    'SamplesPerFrame', frameLength);

info(rx);

disp('接收脚本已启动！正在连续接收并显示频谱...');
disp('按 Ctrl+C 停止接收。');

frameCount = 0;

%% ====== 连续接收循环 ======
try
    while true
        [rxSig, len, overrun] = rx();
        
        frameCount = frameCount + 1;
        
        if overrun
            warning('接收 overrun');
        elseif len == 0
            fprintf('第 %d 帧无数据\n', frameCount);
            continue;
        end
        
        % 每 20 帧显示一次频谱
        if mod(frameCount, 20) == 0
            figure('Name', sprintf('接收频谱 - 第 %d 帧', frameCount));
            pwelch(rxSig, [], [], [], sampleRate, 'centered');
            title(sprintf('接收信号频谱 (设备 %s) - 第 %d 帧', ipAddress, frameCount));
            xlabel('频率 (Hz)'); ylabel('功率谱密度 (dB/Hz)');
            grid on;
        end
        
        pause(0.01);
    end
catch ME
    if ~strcmp(ME.identifier, 'MATLAB:interrupt')
        rethrow(ME);
    end
end

%% ====== 释放资源 ======
release(rx);
disp('接收停止，资源已释放。');