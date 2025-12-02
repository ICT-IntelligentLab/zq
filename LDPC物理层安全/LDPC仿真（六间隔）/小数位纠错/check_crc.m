function is_valid = check_crc(rx_bits, poly_info)
    rx_bits = double(rx_bits(:)); % 保证列向量
    poly = poly_info.poly;
    data_len = poly_info.data_len;
    crc_len = poly_info.crc_len;

    % 拆分数据和CRC
    data_bits = rx_bits(1:data_len);
    recv_crc  = rx_bits(data_len+1 : data_len+crc_len);

    % 重新计算CRC
    calc_crc_bits = calc_crc(data_bits, poly);

    % 比较（都为列向量）
    is_valid = isequal(calc_crc_bits, recv_crc);
end