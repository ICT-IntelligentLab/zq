% 生成 0~9 的 4位格雷码
function bin_str=decimal_to_4bit_binary(n)
    gray = bitxor(n, bitshift(n, -1)); % 格雷码公式
    bin_str = dec2bin(gray, 4);        % 转成4位二进制字符串
   
end
