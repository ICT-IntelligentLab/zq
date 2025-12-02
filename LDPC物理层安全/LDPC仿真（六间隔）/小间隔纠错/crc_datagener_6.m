function databits=crc_datagener_6(index)

            switch index
            case 1
                 N=5;
            case 2
                N=21;
            case 3
               N=63;
            case 4
                N=84;
            case 5
             N=272;
            case 6
              N=456; 
            end
     databits =randi([0 1],N,1);

end