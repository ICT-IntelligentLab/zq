function small_inteval=calculate_small_inteval_6(index,CQI)     

      switch index
        case 1
            a=abs(CQI-0);
            x=a/0.1823;        
        case 2
            a=abs(CQI-0.1823);
            x=a/0.2232; 
        case 3
            a=abs(CQI-0.4055);
            x=a/0.2876;
        case 4
            a=abs(CQI-0.6931);
            x=a/0.4055;
        case 5
            a=abs(CQI-1.0986);
            x=a/0.6932;
        case 6
            a=abs(CQI-1.7918);
            x=a/1.4;
            if x>0.9999
               x=0.999;
            end
      end
      small_inteval=fix(x*8);
      if small_inteval>=8
         small_inteval=7;
      end
end