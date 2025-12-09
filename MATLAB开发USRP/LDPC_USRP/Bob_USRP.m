%% B端：
clear; clc;

%% ========== 系统参数 ==========
Fs   = 4e6;     % 发/收采样率
Rs   = 1e6;      % 符号率
sps  = Fs / Rs;  % 每符号采样数 = 4


%% ========== Mod / RRC ==========
rrcSpan = 8;
rrcTx = rcosdesign(0.25, rrcSpan, sps, 'sqrt');
rrcRx = rrcTx;

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

% 保护polit
txWave = [zeros(rrcSpan,1); pilot_Wave; zeros(rrcSpan,1)];  % 补尾确保完整

%% ========== 首次主动发送导频 ==========
fprintf('B：发送首次导频...\n');
tx(txWave);
pause(0.01);

%% ========== 主循环 ==========
while true  

    rxRaw = [];
    while isempty(rxRaw)
        rxRaw = rx();
    end

   % -------- 用 PreambleDetector检测pilot ----------
    [~, detMetric] = detector(rxRaw);   

    % 取出polit末尾索引
    [~,startSym] = max(detMetric);
    
     if maxnum> mean(detMetric)*5
        if startSym>7443
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
        ss= rxRaw(startSym-88 : startSym+660);
    
        % -------- 匹配滤波 + 降采样 得到符号序列 ----------
        mf = upfirdn(ss, rrcRx, 1, sps);   
    
        % 去除 RRC group delay
        rxSyms = mf(rrcSpan+1 : end-rrcSpan);  % 每个元素对应一个符号
        rx_pilot=rxSyms(1:15);
        rx_data=rxSyms(16:175);
        rx_check=rxSyms(176:180);
    
        fprintf('B：已收到数据（pilot=%d, data=%d, check=%d）\n', ...
            length(rx_pilot), length(rx_data), length(rx_check));
    
        %% ====信道估计与纠错=====
        h_b = sum(rx_pilot .* conj(pilot)) / sum(abs(pilot).^2);

        CQI = abs(h_b)^2;
        index = calculate_interval_10(CQI);
        noiseVar = mean(abs(rx_pilot-h_b*pilot));
        b_index = calculate_small_inteval_10(index,CQI);   
    
        H=  load('H16.mat'); 
        H = H.H;
        H = sparse(H ~= 0);
 
        %qpsk解调
        demodSignal = qam64demod_1(rx_check,h_b,noiseVar,'llr');

        %进行LDPC解码
         maxnumiter=20;
         cfgLDPCDec = ldpcDecoderConfig(H);
         rx_bits = ldpcDecode(demodSignal,cfgLDPCDec,maxnumiter);
         str = char(rx_bits + '0');
         a_index = bit3_binary_to_decimal(str');

         if a_index<4 && b_index>a_index+4 && index<10
                        index=index+1;
         end
         if a_index>3 && b_index<a_index-3 && index>1
                        index=index-1;
         end
     %% ====数据解调和解码=====
          u=[index,index-1,index+1];
          u = u(u > 0 & u < 11);

          for i=1:length(u)

              index = u(i);
              H_b = index_to_H_10(index);
              H_b = sparse(H_b ~= 0);

              %qpsk解调
              demodSignal = qam64demod_1(rx_data,h_b,noiseVar,'llr');

              %进行LDPC解码
              maxnumiter=20;
              cfgLDPCDec = ldpcDecoderConfig(H_b);
              rx_bits = ldpcDecode(demodSignal,cfgLDPCDec,maxnumiter);
                
              poly_info=crc_polynomial_output_10(index);
              is_valid = check_crc(rx_bits, poly_info);
              rxbits =rx_bits ;
              
              if is_valid
                  break;
               end
           end

        %% =====处理完后再次发送导频
        tx(txWave);
        fprintf('B：已发送导频...\n');
        pause(0.01);
     end
     rxRaw_temp=rxRaw;
end