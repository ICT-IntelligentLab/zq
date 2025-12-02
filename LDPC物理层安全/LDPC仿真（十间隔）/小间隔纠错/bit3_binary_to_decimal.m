function dec = bit3_binary_to_decimal(gray_str)
    g = gray_str - '0';     % [0 1 1] 之类
    b = zeros(size(g));     % 用来恢复二进制
    b(1) = g(1);            % 最高位相同
    for i = 2:length(g)
        b(i) = xor(b(i-1), g(i));   % 逐位恢复
    end
    dec = bin2dec(char(b + '0'));   % 转为十进制
end
