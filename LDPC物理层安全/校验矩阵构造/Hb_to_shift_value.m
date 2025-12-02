%%将基矩阵转换为移位值矩阵
function H_Shift_value =Hb_to_shift_value(H_base,Z_c)

    H_Shift_value=zeros(size(H_base));
    
    for i=1:size(H_base,1)
        for j=1:size(H_base,2)
            if H_base(i,j)==1        
                H_Shift_value(i,j)=mod(i*j,Z_c);
            else 
                H_Shift_value(i,j)=-1;
            end
        end
    end
    
end