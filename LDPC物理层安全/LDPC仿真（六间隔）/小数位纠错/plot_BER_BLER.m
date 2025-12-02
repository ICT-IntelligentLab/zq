function plot_BER_BLER(SNR,BER,BER_e,BLER,BLER_e)

fprintf('Bob:\n');
for j=1:size(SNR,2)
  fprintf('SNR=%d:\n', SNR(j));
  fprintf('误码率：%f\n', BER(j));  
  fprintf('误块率：%f\n', BLER(j)); 
end
fprintf('Eve:\n');
for j=1:size(SNR,2)
  fprintf('SNR=%d:\n', SNR(j));
  fprintf('误码率：%f\n', BER_e(j));  
  fprintf('误块率：%f\n', BLER_e(j)); 
end

% 绘制 BER vs. SNR 曲线
figure;  % 创建一个新图窗口

% 绘制两条BER曲线，使用不同的线型和标记
semilogy(SNR, BER, '-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'Color', 'b'); % 曲线1：蓝色，圆形标记
hold on; % 保持当前图像，允许继续在图上绘制其他曲线
semilogy(SNR, BER_e, '-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'Color', 'r'); % 曲线2：红色，方形标记

% 添加标题和标签
title('BER vs. SNR');               % 标题
xlabel('Eb/σ_Z^2 [dB]');             % X 轴标签
ylabel('Bit Error Rate');            % Y 轴标签

% 网格线和图例
grid on;                            % 显示网格线
legend('Bob-BER', 'Eve-BER'); % 为两条曲线添加图例
hold off;  % 释放当前图像


% 绘制 BLER vs. SNR 曲线
figure;  % 创建一个新图窗口

% 绘制两条BLER曲线，使用不同的线型和标记
semilogy(SNR, BLER, '-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'Color', 'b'); % 曲线1：蓝色，圆形标记
hold on; % 保持当前图像，允许继续在图上绘制其他曲线
semilogy(SNR, BLER_e, '-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'Color', 'r'); % 曲线2：红色，方形标记

% 添加标题和标签
title('BLER vs. SNR');               % 标题
xlabel('Eb/σ_Z^2 [dB]');             % X 轴标签
ylabel('Block Error Rate');            % Y 轴标签

% 网格线和图例
grid on;                            % 显示网格线
legend('Bob-BLER', 'Eve-BLER'); % 为两条曲线添加图例
hold off;  % 释放当前图像
end