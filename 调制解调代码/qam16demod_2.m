function demodsignal=qam16demod_2(signal,h,noiseVar,outputType)
  
   if noiseVar == 0
        error('noiseVar不能为零');
   end 

  if h == 0
        error('h不能为零');
  end 

  n=length(signal);
  demodsignal=zeros(4*n,1);
  B1=[3,1,-1,-3];
  B2=[3,-3,1,-1];
  A1=zeros(1,4);
  A2=zeros(1,4);
  %计算LLR
  for i=1:n
      y=real(signal(i)/h);
      for j=1:4
          A1(j)=-(y-B1(j))^2/(2*noiseVar);
          A2(j)=-(y-B2(j))^2/(2*noiseVar);
      end
      demodsignal(4*i-3)=1/sqrt(2*pi*noiseVar)*(max(A1(1:2))-max(A1(3:4)));
      demodsignal(4*i-2)=1/sqrt(2*pi*noiseVar)*(max(A2(1:2))-max(A2(3:4)));
       y=imag(signal(i)/h);
      for j=1:4
          A1(j)=-(y-B1(j))^2/(2*noiseVar);
          A2(j)=-(y-B2(j))^2/(2*noiseVar);
      end
      demodsignal(4*i-1)=1/sqrt(2*pi*noiseVar)*(max(A1(1:2))-max(A1(3:4))); 
      demodsignal(4*i)=1/sqrt(2*pi*noiseVar)*(max(A2(1:2))-max(A2(3:4)));
  end

  % 判断输出类型
  if nargin < 4
      outputType = 'llr';  % 如果用户没指定，默认输出bit
  end

  if strcmpi(outputType, 'bit')
      % 硬判决
      for i = 1:4*n
          if demodsignal(i) < 0
              demodsignal(i) = 1;
          else
              demodsignal(i) = 0;
          end
      end
  elseif strcmpi(outputType, 'llr')
      % 什么也不做，保留LLR
  else
      error('outputType必须是"bit"或"llr"');
  end

end