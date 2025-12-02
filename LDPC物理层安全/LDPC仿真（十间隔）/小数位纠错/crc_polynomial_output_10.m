function poly_info=crc_polynomial_output_10(index)

            switch index
            case 1
                poly_info.poly =[1 0 0 1 1];
                poly_info.crc_len = 4;
                poly_info.data_len = 1;
            case 2
                poly_info.poly =[1 0 0 1 1];
                poly_info.crc_len = 4;
                poly_info.data_len = 8;
            case 3
                poly_info.poly =[1 0 0 0 0 0 1 0 1];
                poly_info.crc_len = 8;
                poly_info.data_len = 21;
              
            case 4
                poly_info.poly =[1 0 0 0 0 0 1 0 1];
                poly_info.crc_len = 8;
                poly_info.data_len = 49;
               
            case 5
                poly_info.poly = [1 1 0 0 0 0 0 0 0 1 0 0 1]; 
                poly_info.crc_len = 12;
                poly_info.data_len =65;
            
            case 6
                poly_info.poly = [1 1 0 0 0 0 0 0 0 1 0 0 1]; 
                poly_info.crc_len = 12;
                poly_info.data_len = 84;
             
            case 7
                poly_info.poly =[1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 0 1]; 
                poly_info.crc_len = 16;
                poly_info.data_len = 176;
            
            case 8
                poly_info.poly = [1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 0 1]; 
                poly_info.crc_len = 16;
                poly_info.data_len = 272;
            
            case 9
                poly_info.poly =  [1 1 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 0 0 0 1 1 1 0];
                poly_info.crc_len = 24;
                poly_info.data_len =360;
             
            case 10
                poly_info.poly = [1 1 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 0 0 0 1 1 1 0];
                poly_info.crc_len = 24;
                poly_info.data_len =456;
            
            end
end
