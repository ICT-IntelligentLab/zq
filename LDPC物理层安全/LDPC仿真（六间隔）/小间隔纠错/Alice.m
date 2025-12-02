function [modSignal,noisedsignal,data_bits,h,g,noiseVar,a_small_inteval]=Alice(SNR,R)

            h1= 1/sqrt(2)*(randn + 1i*randn);  
            g1= 1/sqrt(2)*(randn + 1i*randn); 

            A=sqrt(R)*[h1;g1];
            h=A(1);
            g=A(2);

            pilot_symbols = [  1+1i,  1-1i,  -1+1i, -1-1i];
            pilot = repmat(pilot_symbols, 1, 64/4); % 长度为 N

            noiseVar=2/(10^(SNR/10));
            noise_std = sqrt(noiseVar/2);
            noise_a = noise_std * (randn(1, 64) + 1j * randn(1, 64));

            % 接收信号
            rx_a = h * pilot + noise_a;
            h_a = sum(rx_a .* conj(pilot)) / sum(abs(pilot).^2);

            CQI = abs(h_a)^2;
    
            [index,H]=choose_H_6(CQI);
            
            %将每个间隔再分为8个小间隔，计算估计值在那个小间隔
            a_small_inteval=calculate_small_inteval_6(index,CQI);
         
            %disp(index)
            H = sparse(H ~= 0);
        
            %生成原始数据
            cfgLDPCEnc = ldpcEncoderConfig(H);
            %databits =randi([0 1],cfgLDPCEnc.NumInformationBits,1,'int8');

            %不同间隔需要不同CRC位数，需要的数据位数也不同
            databits=crc_datagener_6(index);

            %添加CRC
            [data_bits, ~] = add_crc(databits);

            %进行LDPC编码
            codeword = ldpcEncode(data_bits, cfgLDPCEnc);
        
            %qpsk调制  
            modSignal = qpskmod(codeword);
           
            %经过衰落信道
            chanSignal=h*modSignal;
         
            %加噪 
            noise = sqrt(noiseVar/2) * (randn(length(chanSignal), 1) + 1i * randn(length(chanSignal), 1));  
            noisedsignal = chanSignal+noise;
end