%% A端：
clear; clc;

%% ========== 系统参数 ==========
Fs   = 4e6;         % 采样率（总采样）
Rs   = 1e6;          % 符号率
sps  = Fs / Rs;      % 每符号采样数 = 4

      
%% ========== RRC ==========
rrcSpan = 8; % RRC span in symbols
rrcTx = rcosdesign(0.25, rrcSpan, sps, 'sqrt');  % 发射端 pulse shape
rrcRx = rrcTx;                                    % 匹配滤波器

%% ========== USRP ==========
rx = sdrrx('N321','IPAddress','192.168.30.3');
rx.CenterFrequency = 3.4e9;
rx.BasebandSampleRate = Fs;
rx.Gain = 20;
rx.SamplesPerFrame = 8192;         
rx.OutputDataType = 'double';         % complex double 输出
rx.EnableBurstMode = false;
rx.Timeout = 1; 

tx = sdrtx('N321','IPAddress','192.168.30.3');
tx.CenterFrequency = 3.4e9;
tx.BasebandSampleRate = Fs;
tx.Gain = 20;
tx.ShowAdvancedProperties = true;
tx.EnableBurstMode = false;

%% ========== 已知的 pilot ==========
pilot = randi([0 1], 90, 1);
pilot_syms = qam64mod(pilot);

% RRC 成型并上采样为发送采样率
pilot_Wave = upfirdn(pilot_syms, rrcTx, sps);  

% Detector
detector = comm.PreambleDetector(pilot_Wave);  

rxRaw_temp=[];
%% ========== 主循环 ==========
while true

    % -------- 从 USRP 拉取采样流（采样率 = Fs） ----------
    rxRaw = [];
    while isempty(rxRaw)
        rxRaw = rx();
    end

    % -------- 用 PreambleDetector检测pilot ----------
    [~, detMetric] = detector(rxRaw);   

    % 取出polit末尾索引
    [maxnum,startSym] = max(detMetric); 
    
    if maxnum> mean(detMetric)*5
        if startSym==8192
            rxRaw1 = [];
            while isempty(rxRaw1)
                    rxRaw1 = rx();
            end
            rxRaw_sum=[rxRaw;rxRaw1];

            % -------- 用 PreambleDetector检测pilot ----------
            [~, detMetric] = detector(rxRaw_sum);   

            % 取出polit末尾索引
            [~,startSym] = max(detMetric); 
            
        elseif startSym<89

            rxRaw_sum=[rxRaw_temp;rxRaw];

            % -------- 用 PreambleDetector检测pilot ----------
            [~, detMetric] = detector(rxRaw_sum);   

            % 取出polit末尾索引
            [~,startSym] = max(detMetric); 
            
        end
        ss= rxRaw(startSym-88 : startSym);
    
        % -------- 匹配滤波 + 降采样 得到符号序列 ----------
        mf = upfirdn(ss, rrcRx, 1, sps);    
    
        % 去除 RRC group delay
        rx_pilot = mf(rrcSpan+1 : end-rrcSpan);  % 每个元素对应一个符号
        fprintf('A：已收到导频（pilot=%d）\n', length(rx_pilot));

      %% ======信道估计=======
         h_a = sum(rx_pilot .* conj(pilot)) / sum(abs(pilot).^2);

         CQI = abs(h_a)^2;
  
         %将每个间隔再分为8个小间隔，计算估计值在那个小间隔
         a_index=calculate_small_inteval_10(index,CQI);
         
         % 转为格雷码（二进制字符串）
         gray1 = decimal_to_3bit_binary(a_index);
         gray_vec = gray1-'0'; 

         H=  load('H16.mat'); 
         H = H.H;
         H = sparse(H ~= 0);

         cfgLDPCEnc = ldpcEncoderConfig(H);

         cherckword = ldpcEncode(gray_vec', cfgLDPCEnc); 
    
    
       %% ===== 生成数据并发送给 B ====

          [index,H]=choose_H_10(CQI);
          H = sparse(H ~= 0);
        
          %生成原始数据
          cfgLDPCEnc = ldpcEncoderConfig(H);

          %不同间隔需要不同CRC位数，需要的数据位数也不同
          databits=crc_datagener_10(index);

          %添加CRC
          [data_bits, ~] = add_crc(databits);

          %进行LDPC编码
          codeword = ldpcEncode(data_bits, cfgLDPCEnc);
        
          data_sum=[pilot;codeword;cherckword];
          data_syms = qam64mod(data_sum);

          % RRC 成型并上采样为发送采样率
          tx_Wave = upfirdn(data_syms, rrcTx, sps);  % 产生采样流 
        
          %在前后补零以保护滤波
          txWave = [zeros(rrcSpan,1); tx_Wave; zeros(rrcSpan,1)]; 
          tx(txWave);
          fprintf('A：已发送数据...');
          pause(0.01);
    end
    rxRaw_temp=rxRaw;
end