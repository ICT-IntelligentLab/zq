function index=calculate_interval_6(CQI)

        if CQI < 0.1823
              index=1;
        elseif CQI >= 0.1823 && CQI < 0.4055
              index=2;
        elseif CQI >= 0.4055 && CQI < 0.6931
              index=3;
        elseif CQI >= 0.6931 && CQI < 1.0986
              index=4;
        elseif CQI >= 1.0986 && CQI < 1.7918
              index=5;
        elseif CQI >=1.7918 
            index=6;
        end
end