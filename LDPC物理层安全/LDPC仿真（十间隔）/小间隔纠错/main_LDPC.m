delete(gcp('nocreate'));
run =10000;

%Np=2;
SNR = [6,8,10,12,14,16];
%var=1./(Np*10.^(SNR./10));

BER=zeros(1,size(SNR,2));
BLER=zeros(1,size(SNR,2));
BER_e=zeros(1,size(SNR,2));
BLER_e=zeros(1,size(SNR,2));
ber = comm.ErrorRate;
R=[1 0.8;
   0.8 1];

p=parpool(6);
parfor i=1:6
    a=0;
    t=0; 
    %data=[];
    %datax=[];
    %data_e=[];

    for n =1:run

        %Alice发射信号并在信道中传播
        [modSignal,noisedsignal,databits,h,g,noiseVar,a_small_inteval]=Alice(SNR(i),R);

        %Bob接收信号并解码
        rxbits=Bob(noisedsignal,noiseVar,h,a_small_inteval); 
           
        %Eve窃听信号并解码
        rxbits_e=Eve(modSignal,noiseVar,g,a_small_inteval);

        %计算Bob误块率
        if length(rxbits)~=length(databits)
           t=t+1;
        else
            reset(ber);
            p_e = ber(databits,double(rxbits));
            if p_e(1)~=0
                t=t+1;
            end
        end

        %计算Eve误块率
        if length(rxbits_e)~=length(databits)
           a=a+1;
        else
            reset(ber);
            p_e = ber(databits,double(rxbits_e));
            if p_e(1)~=0
                a=a+1;
            end
        end

        %data=cat(1,data,databits);
        %datax=cat(1,datax,rxbits);
        %data_e=cat(1,data_e,rxbits_e);

    end


    % %计算Bob误码率
    % if length(datax)>length(data)
    %    datax=datax(1:length(data));
    % elseif length(datax)<length(data)
    %        dt=1-data((length(datax)+1):end);
    %        datax=cat(1,datax,dt);
    % end
    % reset(ber);
    % q=ber(data,datax);
    % BER(i)=q(1);
     BLER(i)= t/run;

    % %计算Eve误码率
    % if length(data_e)>length(data)
    %    data_e=data_e(1:length(data));
    % elseif length(data_e)<length(data)
    %        dt=1-data((length(data_e)+1):end);
    %        data_e=cat(1,data_e,dt);
    % end
    % reset(ber);
    % q_e=ber(data,data_e);
    % BER_e(i)=q_e(1);
    BLER_e(i)= a/run;

end

plot_BER_BLER(SNR,BER,BER_e,BLER,BLER_e);
delete(gcp('nocreate'));


