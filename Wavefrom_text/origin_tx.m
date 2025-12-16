clear; clc;

%% ====== 发射参数 ======
ipAddress       = '192.168.10.2';   % 这台电脑连接的 N321 IP（电脑 A 控制的发射设备）
centerFreq      = 3.4e9;            % 中心频率（两台设备必须完全相同）
gainTx          = 40;               % 发射增益（30~70，根据距离/衰减调整）
masterClockRate = 200e6;            % 主时钟率（两台必须相同）
sampleRate      = 10e6;             % 基带采样率
interpFactor    = masterClockRate / sampleRate;  % 20
frameLength     = 1e5;              % 每帧样本数（越大 underrun 越少）
toneFreq        = 1e6;              % 发送单音频率 1 MHz

% 生成发送信号（实部单音）
t = (0:frameLength-1)' / sampleRate;
%txSig = 0.9 * sign(2*pi**t);  % 幅度稍小于1，避免剪切
txSig = 0.8 * square(2 * pi * 0.5e3 * t);
%% ====== 创建发射器 ======
tx = comm.SDRuTransmitter(...
    'Platform', 'N320/N321', ...
    'IPAddress', ipAddress, ...
    'CenterFrequency', centerFreq, ...
    'Gain', gainTx, ...
    'MasterClockRate', masterClockRate, ...
    'InterpolationFactor', interpFactor, ...
    'ChannelMapping', 1);       % 使用 TX/RX 端口发送

info(tx);

disp('发射脚本已启动！正在连续发送 1 MHz 单音信号...');
disp('按 Ctrl+C 停止发送。');

%% ====== 新增：帧计数器 ======
frameCount = 0;

%% ====== 连续发送循环（新增时域图显示）======
try
    while true
        underrun = tx(txSig);
        frameCount = frameCount + 1;  % 帧计数加1
        
        if underrun > 0
            warning('发送 underrun: %d 样本丢失', underrun);
        end
        
        if mod(frameCount, 100) == 0
            figure('Name', sprintf('接收时域波形 - 第 %d 帧', frameCount));
            
            % 计算时间轴
            Ts = 1 / sampleRate;                  % 采样间隔
            t = (0:length(txSig)-1) * Ts;         % 时间向量（秒）
            
            plot(t * 1e3, real(txSig), 'b', 'LineWidth', 1.2);  % 实部（I路）
            hold on;
            plot(t * 1e3, imag(txSig), 'r', 'LineWidth', 1.2);  % 虚部（Q路）
            hold off;
            
            title(sprintf('接收信号时域波形 (设备 %s) - 第 %d 帧', ipAddress, frameCount));
            xlabel('时间 (ms)');
            ylabel('幅度');
            legend('I (实部)', 'Q (虚部)');
            grid on;
            axis tight;  % 自动调整坐标轴范围
        end
        % 每 100 帧显示一次发射信号的时域波形图
      
        
        pause(0.01);  % 小延时减轻 CPU 负载
    end
catch ME
    if ~strcmp(ME.identifier, 'MATLAB:interrupt')
        rethrow(ME);
    end
end


%% ====== 释放资源 ======
release(tx);
disp('发射停止，资源已释放。');