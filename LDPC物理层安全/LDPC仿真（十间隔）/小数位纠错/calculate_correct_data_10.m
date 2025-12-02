function interval_corrrect=calculate_correct_data_10(index,CQI)         
            switch index
            case 1
                a=abs(CQI-0);
                x=a/0.1054;
                interval_corrrect=fix(x*100)/100; 
            case 2
                a=abs(CQI-0.1054);
                x=a/0.1177;
                interval_corrrect=fix(x*100)/100; 
            case 3
                a=abs(CQI-0.2231);
                x=a/0.1336;
                interval_corrrect=fix(x*100)/100; 
            case 4
                a=abs(CQI-0.3567);
                x=a/0.1541;
                interval_corrrect=fix(x*100)/100; 
            case 5
                a=abs(CQI-0.5108);
                x=a/0.1823;
                interval_corrrect=fix(x*100)/100; 
            case 6
                a=abs(CQI-0.6931);
                x=a/0.2232;
                interval_corrrect=fix(x*100)/100; 
            case 7
                a=abs(CQI- 0.9163);
                x=a/0.2877;
                interval_corrrect=fix(x*100)/100; 
            case 8
                a=abs(CQI-1.2040);
                x=a/0.4054;
                interval_corrrect=fix(x*100)/100; 
            case 9
                a=abs(CQI-1.6094);
                x=a/0.6932;
                interval_corrrect=fix(x*100)/100;     
            case 10
                a=abs(CQI-2.3026);
                x=a/1.4;
                if x>0.9999
                  x=0.999;
                end
                interval_corrrect=fix(x*100)/100; 
            end
end