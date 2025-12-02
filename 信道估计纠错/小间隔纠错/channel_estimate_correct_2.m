% 参数设置
N = 64;                  % 导频长度
m = 10000;              % 样本数量
SNR_dB =10;            % SNR in dB
SNR = 10^(SNR_dB / 10); % 线性信噪比

% 导频符号序列
pilot_symbols = [ 1,  1,  1, 1];
pilot = repmat(pilot_symbols, 1, N/4); % 长度为 N

% 初始化误差累加器
mse_a = 0;      % Alice vs h
mse_b = 0;      % Bob vs h
mse_ab = 0;     % Alice vs Bob
a_interval=zeros(m,1);
b_interval=zeros(m,1);
e_interval=zeros(m,1);
interval_true = zeros(m, 1); 
h_a = zeros(m, 1); % 存储Alice的原始信道估计
h_b = zeros(m, 1);
g_e = zeros(m, 1);% 存储Bob的原始信道估计
for i = 1:m
    % 生成瑞利信道 h（复高斯）

    h= 1/sqrt(2)*(randn + 1i*randn);  
    g= 1/sqrt(2)*(randn + 1i*randn); 
    R=[1 0.8;
       0.8 1];
    A=sqrt(R)*[h;g];
    h=A(1);
    g=A(2);
    h_abs=abs(h)^2;
    interval_true(i) =calculate_interval_10(h_abs); 

    % 噪声
    noise_std = sqrt(1 / (2 * SNR));
    noise_a = noise_std * (randn(1, N) + 1j * randn(1, N));
    noise_b = noise_std * (randn(1, N) + 1j * randn(1, N));
    noise_e = noise_std * (randn(1, N) + 1j * randn(1, N));
    
    % 接收信号
    rx_a = h * pilot + noise_a;
    rx_b = h * pilot + noise_b;
    rx_e = g * pilot + noise_e;

    % LS 估计
    ha_temp = sum(rx_a .* conj(pilot)) / sum(abs(pilot).^2);
    hb_temp = sum(rx_b .* conj(pilot)) / sum(abs(pilot).^2);
    ge_temp = sum(rx_e .* conj(pilot)) / sum(abs(pilot).^2);


    h_a(i)=ha_temp;
    h_b(i)=hb_temp;
    g_e(i)=ge_temp;

    ha_abs_array=abs(ha_temp)^2;
    hb_abs_array=abs(hb_temp)^2;
    ge_abs_array=abs(ge_temp)^2;

    a_interval(i)=calculate_interval_10(ha_abs_array);
    b_interval(i)=calculate_interval_10(hb_abs_array);
    e_interval(i)=calculate_interval_10(ge_abs_array);

    % 累积 MSE
    mse_a = mse_a + abs(ha_temp - h)^2/abs(h)^2;
    mse_b = mse_b + abs(hb_temp - h)^2/abs(h)^2;
    mse_ab =mse_ab + abs(ha_temp - hb_temp)^2/abs(hb_temp)^2;
end

% 求平均
MSE_Alice = mse_a / m;
MSE_Bob = mse_b / m;
MSE_Alice_Bob = mse_ab / m;


diff_count = sum(a_interval ~= b_interval);
disp(diff_count)

diff_count1 = sum(a_interval ~= e_interval);
disp(diff_count1)

ha_abs_array=zeros(1,m);
hb_abs_array=zeros(1,m);
ge_abs_array=zeros(1,m);

a_small_inteval=zeros(1,m);
b_small_inteval=zeros(1,m);
e_small_inteval=zeros(1,m);

for i=1:m
    ha_abs_array(i)=abs(h_a(i))^2;
    hb_abs_array(i)=abs(h_b(i))^2;
    ge_abs_array(i)=abs(g_e(i))^2;

    
    a_small_inteval(i)=calculate_small_inteval_10(a_interval(i),ha_abs_array(i));
    b_small_inteval(i)=calculate_small_inteval_10(b_interval(i),hb_abs_array(i)); 
    e_small_inteval(i)=calculate_small_inteval_10(e_interval(i),ge_abs_array(i)); 

    if a_small_inteval(i)<4 && b_small_inteval(i)>a_small_inteval(i)+4 && b_interval(i)<10
                 b_interval(i)=b_interval(i)+1;
    end
    if a_small_inteval(i)>3 && b_small_inteval(i)<a_small_inteval(i)-3 && b_interval(i)>1
                 b_interval(i)=b_interval(i)-1;
    end

    if a_small_inteval(i)<4 && e_small_inteval(i)>a_small_inteval(i)+4 && e_interval(i)<10
                 e_interval(i)=e_interval(i)+1;
    end
    if a_small_inteval(i)>3 && e_small_inteval(i)<a_small_inteval(i)-3 && e_interval(i)>1
                 e_interval(i)=e_interval(i)-1;
    end

end

diff_count2 = sum(a_interval ~= b_interval);
disp(diff_count2)
diff_count3 = sum(a_interval ~= e_interval);
disp(diff_count3)

fprintf('MSE_Alice       = %.6f\n', MSE_Alice);
fprintf('MSE_Bob         = %.6f\n', MSE_Bob);
fprintf('MSE_Alice_Bob   = %.6f\n', MSE_Alice_Bob);
