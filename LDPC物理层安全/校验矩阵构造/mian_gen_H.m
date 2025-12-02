%%使用PEG算法生成不同码率的矩阵

    %基本参数
    %r=[5/960   12/960  29/960  57/960  77/960  0.1  0.2  0.3  0.4  0.5];
    r=[9/960   29/960  75/960    0.1    0.3    0.5];
 
    %基矩阵码长
    n=960;
  
    %基矩阵列重
    %dv=[191   79   33   30   21  16   10   8   6  5];
    dv=[106    33   21     16      8     5];
   
    %扩展因子
    Z_c=[1 1 1 1 1 1 1 1 1 1];
    
for i=1:6
    k=n*r(i);
    m=n-k;
    b=1;
    while(b)
        %生成基矩阵
        H_base=gen_H_base(m,k,dv(i));

        %检测四循环
        [cycle4num,~]=check4cycle(H_base);

        %检测六循环
        [cycle6num,~]=check6cycle(H_base);

        if cycle4num==0&&cycle6num==0      
            b=0;
        end
    end
    fprintf('矩阵%d：\n',i);
    fprintf('存在%d个四循环\n', cycle4num);
    fprintf('存在%d个六循环\n', cycle6num);  
   
    H_Shift_value=Hb_to_shift_value(H_base,Z_c(i));
    Ht=shift_value_to_H( H_Shift_value,Z_c(i));
    P=eye(size(Ht,1));
    H=[Ht,P];
    %生成矩阵名称 
    H_Name = sprintf('H%d',i);
    % 将生成的矩阵保存到对应的 .mat 文件中
    save([H_Name, '.mat'], 'H');
    disp(size(H));
    fprintf('校验矩阵H%d已保存\n',i);
   
end






