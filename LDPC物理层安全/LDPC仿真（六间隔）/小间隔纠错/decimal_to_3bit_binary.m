function bin_str = decimal_to_3bit_binary(n)
           
    gray = bitxor(n, bitshift(n, -1));   % 格雷码
    bin_str = dec2bin(gray, 3);   % 输出 3 位字符串
end
