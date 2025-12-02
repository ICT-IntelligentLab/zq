function H=index_to_H_6(index)

            switch index
            case 1
                 H= load('H1.mat'); 
                 H = H.H;
            case 2
                H= load('H2.mat'); 
                 H = H.H;
            case 3
               H= load('H3.mat'); 
                 H = H.H;
            case 4
                H= load('H4.mat'); 
                 H = H.H;
            case 5
               H= load('H5.mat'); 
                 H = H.H;
            case 6
               H= load('H6.mat'); 
                 H = H.H;
            end
 

end