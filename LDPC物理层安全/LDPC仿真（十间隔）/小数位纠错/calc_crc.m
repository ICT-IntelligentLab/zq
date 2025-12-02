function crc = calc_crc(data_bits, poly)
    % data_bits: 列向量
    % poly: 例如 [1 0 0 1 1] 表示 x^4 + x + 1

    data_bits = double(data_bits(:))';
    poly = double(poly(:))';

    m = length(poly) - 1; % CRC长度
    msg = [data_bits zeros(1, m)]; % 补零准备除法

    % 模2除法
    for i = 1:length(data_bits)
        if msg(i) == 1
            msg(i:i+m) = xor(msg(i:i+m), poly);
        end
    end

    crc = msg(end-m+1:end)'; % 取余数部分
end

