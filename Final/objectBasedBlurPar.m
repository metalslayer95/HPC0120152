clc;	% Clear command window.
%clear;	% Delete all variables.
imtool close all;	% Close all figure windows created by imtool.
gd = gpuDevice();
reset(gd); % vaciar memoria usada gpu
I =imread('imagenesPrueba/imgPrueba1.jpg');
Igpu = gpuArray(I);
auxIgpu = Igpu;
[n,m,ch] = size(auxIgpu);
Isize = gpuArray([n,m,ch]);
nFilters = 3;
%%%
Ibwgpu = rgb2gray(Igpu);
ind = find(Ibwgpu < 165);
ind2 = find(Ibwgpu >= 165);
Ibwgpu(ind) = 0;
Ibwgpu(ind2) = 255;

Ibw = gather(Ibwgpu); % se pasa nuevamente a memoria de CPU para ejecutar bwperim
b = bwperim(Ibw,8);  % se encuentra el perimetro de los objetos en la imagen
[B,L] = bwboundaries(b,'holes');  %Agujeros negros
Lgpu = gpuArray(L);
fillgpu= imfill(Lgpu,'holes');          %Lenar agujeros
Ibwgpu = imfill(fillgpu,'holes');
figure('name','Mascara objeto','numberTitle','off')
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
motionFilter = gpuArray(fspecial('motion',20,270));
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

figure('name','Imagen original','numberTitle','off')
imshow(I)

figure('name','Imagen con disk blur','numberTitle','off')
imshow(diskBlur)

figure('name','Imagen con motion blur','numberTitle','off')
imshow(motionBlur)

figure('name','Imagen con gaussian','numberTitle','off')
imshow(gaussianBlur)

figure('name','Imagen con average','numberTitle','off')
imshow(averageBlur)