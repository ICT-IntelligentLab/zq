function index=correct_data_channel_1(h,h_b,noiseVar,ha_index)

            d1 = floor(ha_index * 10);           % 第一位
            d2 = floor(ha_index * 100) - d1*10;  % 第二位

            %转为格雷码（二进制字符串）
            gray1 = decimal_to_4bit_binary(d1);
            gray2 = decimal_to_4bit_binary(d2);

            % 转为数值向量并拼接
            gray_vec = [gray1-'0', gray2-'0']; 

            H = load('H15.mat'); 
            H = H.H;
            H = sparse(H ~= 0);

            cfgLDPCEnc = ldpcEncoderConfig(H);

            codeword = ldpcEncode(gray_vec', cfgLDPCEnc);
        
            %qpsk调制  
            modSignal = qpskmod(codeword);

            %经过衰落信道
            chanSignal=h*modSignal;
         
            %加噪 

            noise = sqrt(noiseVar/2) * (randn(length(chanSignal), 1) + 1i * randn(length(chanSignal), 1));  
            noisedsignal = chanSignal+noise;

            %qpsk解调
            demodSignal = qpskdemod(noisedsignal,h_b,noiseVar,'llr');

            %进行LDPC解码
            maxnumiter=20;
            cfgLDPCDec = ldpcDecoderConfig(H);
            rx_bits = ldpcDecode(demodSignal,cfgLDPCDec,maxnumiter);

            str = char(rx_bits + '0');
            group1 = (str(1:4))';
            group2 = (str(5:8))';

         dec1 = bit4_binary_to_decimal(group1);
         dec1 = mod(dec1, 10); 
         dec2 = bit4_binary_to_decimal(group2);
         dec2 = mod(dec2, 10); 
         index = dec1*0.1 + dec2*0.01;
  
end


