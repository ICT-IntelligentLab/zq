function rxbits_e=Eve(modSignal,noiseVar,g,a_small_inteval_temp)

        %经过衰落信道
        chanSignal_e=g*modSignal;

        %加噪             
        noise_e = sqrt(noiseVar/2) * (randn(length(chanSignal_e), 1) + 1i * randn(length(chanSignal_e), 1));    
        noisedsignal_e = chanSignal_e+noise_e;

        pilot_symbols = [  1+1i,  1-1i,  -1+1i, -1-1i];
        pilot = repmat(pilot_symbols, 1, 64/4); % 长度为 N

        noise_std = sqrt(noiseVar/2);
        noise_e1 = noise_std * (randn(1, 64) + 1j * randn(1, 64));

        % 接收信号
        rx_e = g * pilot + noise_e1;
        g_e = sum(rx_e .* conj(pilot)) / sum(abs(pilot).^2);
         
        CQI = abs(g_e)^2;
        index=calculate_interval_10(CQI);

        %纠错方法二
        e_small_inteval=calculate_small_inteval_10(index,CQI);

        a_small_inteval=correct_data_channel_2(g,g_e,noiseVar,a_small_inteval_temp);
       
        if a_small_inteval<4 && e_small_inteval>a_small_inteval+4 && index<10
                        index=index+1;        
         end
         if a_small_inteval>3 && e_small_inteval<a_small_inteval-3 && index>1      
                        index=index-1;         
         end

        u=[index,index-1,index+1,index-2,index+2];
        u = u(u > 0 & u < 11);

        for i=1:length(u)
           
           index=u(i);
           H_e=index_to_H_10(index);
           H_e = sparse(H_e ~= 0);
                                
           %qpsk解调
           demodSignal_e = qpskdemod(noisedsignal_e,g_e,noiseVar,'llr');

           maxnumiter=20;
           cfgLDPCDec = ldpcDecoderConfig(H_e);
           rx_bits = ldpcDecode(demodSignal_e,cfgLDPCDec,maxnumiter);

           poly_info=crc_polynomial_output_10(index);
           is_valid = check_crc(rx_bits, poly_info);
           rxbits_e =rx_bits ;
           if is_valid
               break;
           end
        end
end