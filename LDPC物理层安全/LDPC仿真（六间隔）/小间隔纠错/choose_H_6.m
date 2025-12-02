function [index,H]=choose_H_6(CQI)

      if CQI < 0.1823
          H= load('H1.mat'); 
          H = H.H;
          index=1;
        elseif CQI >= 0.1823 && CQI < 0.4055
            H= load('H2.mat'); 
            H = H.H;
              index=2;
        elseif CQI >= 0.4055 && CQI < 0.6931
            H= load('H3.mat'); 
            H = H.H;
              index=3;
        elseif CQI >= 0.6931 && CQI < 1.0986
            H= load('H4.mat'); 
            H = H.H;
              index=4;
        elseif CQI >= 1.0986 && CQI < 1.7918
            H= load('H5.mat'); 
            H = H.H;
              index=5;
        elseif CQI >=1.7918 
            H= load('H6.mat'); 
            H = H.H;
            index=6;
      end
end