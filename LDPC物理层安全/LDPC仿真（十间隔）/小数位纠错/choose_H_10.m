function [index,H]=choose_H_10(CQI)

      if CQI < 0.1054
          H= load('H1.mat'); 
          H = H.H;
          index=1;
        elseif CQI >= 0.1054 && CQI < 0.2231
              H=  load('H2.mat'); 
              H = H.H;
              index=2;
        elseif CQI >= 0.2231 && CQI < 0.3567
              H=  load('H3.mat');
              H = H.H;
              index=3;
        elseif CQI >= 0.3567 && CQI < 0.5108
              H=  load('H4.mat'); 
              H = H.H;
              index=4;
        elseif CQI >= 0.5108 && CQI < 0.6931
              H=  load('H5.mat'); 
              H = H.H;
              index=5;
        elseif CQI >= 0.6931 && CQI < 0.9163
              H= load('H6.mat'); 
              H = H.H;
              index=6;
         elseif CQI >= 0.9163 && CQI < 1.2040
              H=  load('H7.mat'); 
              H = H.H;
              index=7;
        elseif CQI >= 1.2040 && CQI < 1.6094
              H= load('H8.mat'); 
              H = H.H;
              index=8;
        elseif CQI >= 1.6094 && CQI < 2.3026
             H= load('H9.mat'); 
             H = H.H;
             index=9;
        elseif CQI >=2.3026 
            H= load('H10.mat'); 
            H = H.H;
            index=10;
      end
end