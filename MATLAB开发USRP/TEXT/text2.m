%% B测试
clear; clc;

%% ========== 系统参数 ==========
Fs   = 4e6;         % 采样率（总采样）
Rs   = 1e6;          % 符号率
sps  = Fs / Rs;      % 每符号采样数 = 4
frameLen = 180;


%% ========== Mod / RRC ==========
rrcSpan = 8;   % RRC span in symbols
rrcTx = rcosdesign(0.25, rrcSpan, sps, 'sqrt');  % 发射端 pulse shape
rrcRx = rrcTx;                                    % 匹配滤波器


%% ========== 已知的 pilot ==========

pilot_bits = randi([0 1], 90, 1);
pilot_syms = qam64mod(pilot_bits);
disp(pilot_syms)

% RRC 成型并上采样为发送采样率
pilot_Wave = upfirdn(pilot_syms, rrcTx, sps);  

% Detector
detector = comm.PreambleDetector(pilot_Wave);  % 符号级输入模式

%% ========== 主循环 ==========

    data_bits = randi([0 1], 960, 1);
    data_syms = qam64mod(data_bits);
    disp(data_syms)
    crc_bits = randi([0 1], 30, 1);
    crc_syms = qam64mod(crc_bits);
    disp(crc_syms)
    sum_syms=[pilot_syms ;data_syms;crc_syms];

    % RRC 成型并上采样为发送采样率
    Wave = upfirdn(sum_syms, rrcTx, sps); 
    h= 1/sqrt(2)*(randn + 1i*randn);  
    Wave= Wave*h;
    start=200;
    noise = sqrt(1/2) * (randn(8192, 1) + 1i * randn(8192, 1)); 
    noise(start:start + length(Wave) - 1) = noise(start:start + length(Wave) - 1) + Wave;
    rxRaw=noise;
    

    % -------- 用 PreambleDetector检测pilot ----------
    [detPositions, detMetric] = detector(rxRaw);   

    % 取出polit末尾索引
    [w,startSym] = max(detMetric); 
    ss= rxRaw(startSym-88 : startSym+660);
    w1=mean(detMetric);
    % -------- 匹配滤波 + 降采样 得到符号序列 ----------
    mf = upfirdn(ss, rrcRx, 1, sps);    % result is already decimated by sps

    % 去除 RRC group delay（samples），注意边界检查
    rxSyms = mf(rrcSpan+1 : end-rrcSpan);  % 每个元素对应一个符号
    polit=rxSyms(1:15);
    data=rxSyms(16:175);
    check=rxSyms(176:180);
    disp(polit)
    disp(data)
    disp(check)
