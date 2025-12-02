function index=calculate_interval_10(CQI)

        if CQI < 0.1054
              index=1;
        elseif CQI >= 0.1054 && CQI < 0.2231
              index=2;
        elseif CQI >= 0.2231 && CQI < 0.3567
              index=3;
        elseif CQI >= 0.3567 && CQI < 0.5108
              index=4;
        elseif CQI >= 0.5108 && CQI < 0.6931
              index=5;
        elseif CQI >= 0.6931 && CQI < 0.9163
              index=6;
         elseif CQI >= 0.9163 && CQI < 1.2040
              index=7;
        elseif CQI >= 1.2040 && CQI < 1.6094
              index=8;
        elseif CQI >= 1.6094 && CQI < 2.3026
             index=9;
        elseif CQI >=2.3026 
            index=10;
        end
end