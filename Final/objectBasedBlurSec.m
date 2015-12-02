clc;	% Clear command window.
%clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
I = imread('imgPrueba1.jpg');
auxI = I;
nFilters = 3;
[n,m,ch] = size(auxI);
Ibw = rgb2gray(I);
ind = find(Ibw < 165);
ind2 = find(Ibw >= 165);
Ibw(ind) = 0;
Ibw(ind2) = 255;
b = bwperim(Ibw,8); 
[B,L]= bwboundaries(b,'holes');  %Agujeros negros
fill=imfill(L,'holes');          %Llenar agujeros
Ibw = imfill(fill,'holes');
figure(1)
bI = binary(Ibw);
imshow(bI);


%%%% aplicando blur con "disk"
tic
diskFilter = fspecial('disk',5);
diskBlur = objectBlur(I,diskFilter,bI);
timedisk = toc
%%%% aplicando blur con "average"
tic
averageFilter = fspecial('average',[3 3]);
averageBlur = objectBlur(I,averageFilter,bI);
timeaverage = toc

%%% aplicando blur con "motion"
tic
motionFilter = fspecial('motion',5,270);
motionBlur = objectBlur(I,motionFilter,bI);
timemotion = toc

%%% aplicando blur con "gaussian"
tic
gaussianFilter = fspecial('gaussian',5,10);
gaussianBlur = objectBlur(I,gaussianFilter,bI);
timegaussian = toc

figure(2)
%%% graficando imagen original
imshow(I)
title('Original');
figure(3)
% %%% graficando imagen con disk
% %subplot(1,nFilters,1)
imshow(diskBlur)
title('Disk')

figure(4)
% %%% graficando imagen con motion
% subplot(1,nFilters,2)
imshow(motionBlur)
title('Motion')

figure(5)
%%% graficando imagen con gaussian
% subplot(1,nFilters,3)
imshow(gaussianBlur)
title('Gaussian')


figure(6)
%%% graficando imagen con gaussian
% subplot(1,nFilters,3)
imshow(averageBlur)
title('Average')

