% 将移位值矩阵转换为校验矩阵
function H = shift_value_to_H(shift_matrix, Z_c)
    
    [M, N] = size(shift_matrix); % 获取移位值矩阵的尺寸
    H = zeros(M*Z_c, N*Z_c);         % 初始化校验矩阵为零矩阵

    for i = 1:M
        for j = 1:N
            shift_value = shift_matrix(i, j); % 获取移位值
            if shift_value == -1
                % 如果移位值为 -1，生成全零矩阵
                submatrix = zeros(Z_c, Z_c);
            else
                % 否则生成循环移位矩阵
                submatrix = circshift(eye(Z_c), [0, shift_value]);
            end
            % 将生成的子矩阵填充到校验矩阵中
            H((i-1)*Z_c+1:i*Z_c, (j-1)*Z_c+1:j*Z_c) = submatrix;
        end
    end
end
