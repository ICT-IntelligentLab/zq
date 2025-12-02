function [tx_bits, poly_info] = add_crc(data_bits)
    data_bits = double(data_bits(:));  % 确保列向量形式
    n = length(data_bits);

    % 按长度选择CRC多项式（含最高次项）

    if n < 20
        poly =[1 0 0 1 1];
        crc_len = 4;
    elseif n < 50
        poly = [1 0 0 0 0 0 1 0 1];  % CRC-8: x^8 + x^2 + x + 1
        crc_len = 8;
    elseif n < 100
        poly = [1 1 0 0 0 0 0 0 0 1 0 0 1]; % CRC-12: x^12 + x^11 + x^3 + x^2 + x + 1
        crc_len = 12;
    elseif n < 300
        poly = [1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 0 1]; % CRC-16: x^16 + x^15 + x^2 + 1
        crc_len = 16;
    else
        poly = [1 1 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 0 0 0 1 1 1 0]; % CRC-16: x^16 + x^15 + x^2 + 1
        crc_len = 24;
    end

    % 计算CRC校验
    crc_bits = calc_crc(data_bits, poly);
 
    % 拼接发送序列
    tx_bits = [data_bits; crc_bits(:)];

    % 输出CRC信息
    poly_info.poly = poly(:);
    poly_info.crc_len = crc_len;
    poly_info.data_len = n;
end