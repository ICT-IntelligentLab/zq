function index=correct_data_channel_2(h,h_b,noiseVar,ha_index)


            % 转为格雷码（二进制字符串）
            gray1 = decimal_to_3bit_binary(ha_index);
            gray_vec = gray1-'0'; 

            H=  load('H16.mat'); 
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
            dec1 = bit3_binary_to_decimal(str');
            index = dec1;
end


