function  rxbits=Bob(noisedsignal,noiseVar,h, a_small_inteval_temp)         

            %error =1/sqrt(2)*(normrnd(0, sqrt(var)) + 1i * normrnd(0, sqrt(var)));
            %h_b=h+error;

            pilot_symbols = [  1+1i,  1-1i,  -1+1i, -1-1i];
            pilot = repmat(pilot_symbols, 1, 64/4); % 长度为 N

            noise_std = sqrt(noiseVar/2);
            noise_b = noise_std * (randn(1, 64) + 1j * randn(1, 64));

             % 接收信号
             rx_b = h * pilot + noise_b;
             h_b = sum(rx_b .* conj(pilot)) / sum(abs(pilot).^2);


             CQI = abs(h_b)^2;
             index=calculate_interval_10(CQI);
       
             %纠错方法二
             b_small_inteval=calculate_small_inteval_10(index,CQI);

             a_small_inteval=correct_data_channel_2(h,h_b,noiseVar, a_small_inteval_temp);

             if a_small_inteval<4 && b_small_inteval>a_small_inteval+4 && index<10
                        index=index+1;
             end
             if a_small_inteval>3 && b_small_inteval<a_small_inteval-3 && index>1
                        index=index-1;
             end

             u=[index,index-1,index+1];
             u = u(u > 0 & u < 11);

             for i=1:length(u)

                index=u(i);
                H_b=index_to_H_10(index);
                H_b = sparse(H_b ~= 0);

                %qpsk解调
                demodSignal = qpskdemod(noisedsignal,h_b,noiseVar,'llr');

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
end
