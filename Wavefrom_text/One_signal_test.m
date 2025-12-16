clear; clc;
% 参数设置（请根据你的 IP 和需求调整）
ipAddress = '192.168.10.2'; % 替换为你的 N321 IP 地址（默认 192.168.10.2）
centerFreq = 2.4e9; % 中心频率（根据许可频段调整）
gainTx = 30; % 发送增益（N321 范围 0-89.75 dB，慎调避免过载）
gainRx = 30; % 接收增益
masterClockRate = 200e6; % 主时钟率（支持 200e6, 245.76e6, 250e6）
sampleRate = 10e6; % 基带采样率（Interpolation/Decimation = masterClockRate / sampleRate）
interpFactor = masterClockRate / sampleRate; % 插值因子（发送）
decimFactor = masterClockRate / sampleRate; % 抽取因子（接收）
frameLength = 1e5; % 每帧样本数
toneFreq = 1e6;

% 生成发送信号：方波/复正弦/正弦
t = (0:frameLength-1)' / sampleRate;
txSig = square(2*pi*500*t); % 单位幅度
%txSig = cos(2*pi*toneFreq*t);
%txSig = cos(2*1i*pi*toneFreq*t);

tx = comm.SDRuTransmitter(...
    'Platform',            'N320/N321', ...
    'IPAddress',           ipAddress, ...
    'CenterFrequency',     centerFreq, ...
    'Gain',                gainTx, ...
    'MasterClockRate',     masterClockRate, ...
    'InterpolationFactor', interpFactor, ...
    'ChannelMapping',      1);   

rx = comm.SDRuReceiver(...
    'Platform',            'N320/N321', ...
    'IPAddress',           ipAddress, ...
    'CenterFrequency',     centerFreq, ...
    'Gain',                gainRx, ...
    'MasterClockRate',     masterClockRate, ...
    'DecimationFactor',    decimFactor, ...
    'ChannelMapping',      2, ...
    'OutputDataType',      'double', ...
    'SamplesPerFrame',     frameLength);
% 检查设置（可选）
info(tx)
info(rx)
% 连续发送和接收循环（全双工）
for k = 1:200
    underrun = tx(txSig); % 发送（underrun 表示丢失）
if underrun > 0
        warning('发送 underrun: %d 样本丢失', underrun);
end
    [rxSig, len, overrun] = rx(); % 接收
if overrun
        warning('接收 overrun');
elseif len == 0
        fprintf('无数据（可能延迟）\n');
continue;
end
% 简单显示：频谱（应看到 1 MHz 峰值）
% 每 20 帧显示一次发送信号和接收信号的频谱（不覆盖）
if mod(k, 20) == 0
        figure; % 每次创建一个新窗口（不会覆盖旧图）
% 左图：发送信号频谱（理论上应该是干净的单根谱线）
        subplot(1,2,1);
        plot(t * 1e3, real(txSig), 'b', 'LineWidth', 1.2); % 实部（I路）
            hold on;
% 右图：接收信号频谱（实际收到的，应该也有 1 MHz 峰）
        subplot(1,2,2);
        plot(t * 1e3, real(rxSig), 'b', 'LineWidth', 1.2); % 实部（I路）
            hold on;
% 可选：调整整个窗口大小，便于观察
        set(gcf, 'Position', [200, 200, 1200, 400]);
end
end
% 释放资源
release(tx);
release(rx);
disp('结束');