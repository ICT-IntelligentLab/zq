function demodsignal=qam64demod_2(signal,h,noiseVar,outputType)

   if noiseVar == 0
        error('noiseVar不能为零');
   end 

  if h == 0
        error('h不能为零');
  end 

  n=length(signal);
  demodsignal=zeros(6*n,1);
  B1=[7,5,1,3,-7,-5,-1,-3];
  B2=[7,5,-7,-5,1,3,-1,-3];
  B3=[7,1,-7,-1,5,3,-5,-3];
  A1=zeros(1,8);A2=zeros(1,8);A3=zeros(1,8);
  for i=1:n
      y=real(signal(i)/h);
      for j=1:8
          A1(j)=-(y-B1(j))^2/(2*noiseVar);
          A2(j)=-(y-B2(j))^2/(2*noiseVar);
          A3(j)=-(y-B3(j))^2/(2*noiseVar);
      end
      demodsignal(6*i-5)=1/sqrt(2*pi*noiseVar)*(max(A1(1:4))-max(A1(5:8)));
      demodsignal(6*i-4)=1/sqrt(2*pi*noiseVar)*(max(A2(1:4))-max(A2(5:8)));
      demodsignal(6*i-3)=1/sqrt(2*pi*noiseVar)*(max(A3(1:4))-max(A3(5:8)));
       y=imag(signal(i)/h);
       for j=1:8
          A1(j)=-(y-B1(j))^2/(2*noiseVar);
          A2(j)=-(y-B2(j))^2/(2*noiseVar);
          A3(j)=-(y-B3(j))^2/(2*noiseVar);
      end
      demodsignal(6*i-2)=1/sqrt(2*pi*noiseVar)*(max(A1(1:4))-max(A1(5:8)));
      demodsignal(6*i-1)=1/sqrt(2*pi*noiseVar)*(max(A2(1:4))-max(A2(5:8)));
      demodsignal(6*i)=1/sqrt(2*pi*noiseVar)*(max(A3(1:4))-max(A3(5:8)));
  end

  % 判断输出类型
  if nargin < 4
      outputType = 'llr';  % 如果用户没指定，默认输出bit
  end

  if strcmpi(outputType, 'bit')
      % 硬判决
      for i = 1:6*n
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