gd = gpuDevice();
reset(gd);
I =imread('imgPrueba1.jpg');
Igpu = gpuArray(I);
auxIgpu = Igpu;
[n,m,ch] = size(auxIgpu);
Isize = gpuArray([n,m,ch]);
nFilters = 3;
%%%
Ibwgpu = rgb2gray(Igpu);
ind = find(Ibwgpu < 135);
ind2 = find(Ibwgpu >= 135);
Ibwgpu(ind) = 0;
Ibwgpu(ind2) = 255;

Ibw = gather(Ibwgpu); % se pasa nuevamente a memoria de CPU para ejecutar bwperim
b = bwperim(Ibw,8);  % se encuentra el perimetro de los objetos en la imagen
%[b, num]=CapBinaria(Ibw);         %Obtener imagen en FORMATO BINARIO
%b = gpuArray(b);
[B,L] = bwboundaries(b,'holes');  %Agujeros negros
Lgpu = gpuArray(L);
figure(1)
fillgpu= imfill(Lgpu,'holes');          %Lenar agujeros
Ibwgpu = imfill(fillgpu,'holes');
imshow(Ibwgpu);
Ibw = gather(Ibwgpu);


%%%% aplicando blur con "disk"
disp('Empezando disk');
tic
blurFilter = fspecial('disk',5);

% diskBlur = imfilter(Igpu,blurFilter,'replicate');
% diskBlur = gather(diskBlur);
% for i=1:n
%      for j =1:m
%          if (Ibwgpu(i,j) > 0)
%              diskBlur(i,j,:) = auxI(i,j,:);
%          end
%      end
% end
diskBlur = gather(diskBlur);
wait(gd);
timediskGPU = toc
disp('Terminando disk');
pause
disp('Empezando motion');
%%% aplicando blur con "motion"
tic
motionFilter = fspecial('motion',20,45);
motionBlur = imfilter(Igpu,motionFilter,'replicate','conv');
for i=1:n%Isize(1)
     for j =1:m%Isize(2)
         if (Ibwgpu(i,j) > 0)
             motionBlur(i,j,:) = auxIgpu(i,j,:);
         end
     end
      if ((mod(i,10)) == 0)
         disp(i);
      end
end
motionBlur = gather(motionBlur);
wait(gd);
timemotionGPU = toc
disp('Terminando motion');

%%% aplicando blur con "gaussian"
disp('Empezando gaussian');
tic
gaussianFilter = fspecial('gaussian',5,10);
gaussianBlur = imfilter(Igpu,gaussianFilter,'replicate','conv');
for i=1:n%Isize(1)
     for j =1:m%Isize(2)
         if (Ibw(i,j) > 0)
             gaussianBlur(i,j,:) = auxIgpu(i,j,:);
         end
     end
     if ((mod(i,10)) == 0)
         disp(i);
      end
end
gaussianBlur = gather(gaussianBlur);
wait(gd);
timegaussianGPU = toc
disp('Terminando gaussian');

figure(2)
%%% graficando imagen original
imshow(I)
title('Original');
figure(3)
%%% graficando imagen con disk
subplot(1,nFilters,1)
imshow(diskBlur)
title('Disk')

%%% graficando imagen con motion
subplot(1,nFilters,2)
imshow(motionBlur)
title('Motion')

%%% graficando imagen con gaussian
subplot(1,nFilters,3)
imshow(gaussianBlur)
title('Gaussian')

%%%%
% figure(3)
% Ibw = rgb2gray(I);
% ind = find(Ibw < 185);
% ind2 = find(Ibw >= 185);
% Ibw(ind) = 0;
% Ibw(ind2) = 255;
% I2 = bwperim(Ibw,8);
% imshow(I2);
%hold on
%plot(I2);



