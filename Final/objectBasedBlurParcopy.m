clc;	% Clear command window.
%clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.
gd = gpuDevice();
reset(gd); % vaciar memoria usada gpu
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
[B,L] = bwboundaries(b,'holes');  %Agujeros negros
Lgpu = gpuArray(L);
figure(1)
fillgpu= imfill(Lgpu,'holes');          %Lenar agujeros
Ibwgpu = imfill(fillgpu,'holes');
imshow(Ibwgpu);
Ibw = gather(Ibwgpu);
bIgpu = gpuArray(binary(Ibwgpu));
%%%% aplicando blur con "disk"
disp('Empezando disk');
tic
diskFilter = gpuArray(fspecial('disk',5));
diskBlur = objectBlur(Igpu,diskFilter,bIgpu);
diskBlur = gather(diskBlur);
wait(gd);
timediskGPU = toc
disp('Terminando disk');

%%% aplicando blur con average filter
tic
averageFilter = gpuArray(fspecial('average',[3 3]));
averageBlur = objectBlur(Igpu,averageFilter,bIgpu);
averageBlur = gather(averageBlur);
wait(gd);
timeaverageGPU = toc

%%% aplicando blur con "motion"
disp('Empezando motion');
tic
motionFilter = gpuArray(fspecial('motion',20,45));
motionBlur = objectBlur(Igpu,motionFilter,bIgpu);
motionBlur = gather(motionBlur);
wait(gd);
timemotionGPU = toc
disp('Terminando motion');

%%% aplicando blur con "gaussian"
disp('Empezando gaussian');
tic
gaussianFilter = gpuArray(fspecial('gaussian',5,10));
gaussianBlur = objectBlur(Igpu,gaussianFilter,bIgpu);
gaussianBlur = gather(gaussianBlur);
wait(gd);
timegaussianGPU = toc
disp('Terminando gaussian');


%%% graficando

figure(2)
%%% graficando imagen original
imshow(I)
title('Original');

figure(3)
% %%% graficando imagen con disk
imshow(diskBlur)
title('Disk')

figure(4)
% %%% graficando imagen con motion
imshow(motionBlur)
title('Motion')



figure(5)
%%% graficando imagen con gaussian
imshow(gaussianBlur)
title('Gaussian')


figure(6)
%%% graficando imagen con gaussian
imshow(averageBlur)
title('Average')