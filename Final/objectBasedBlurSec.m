clc;	% Clear command window.
%clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
I = imread('imagenesPrueba/imgPrueba1.jpg');
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
figure('name','Mascara objeto','numberTitle','off')
bI = binary(Ibw);
imshow(bI);
imwrite(bI,'resultadosPruebas/maskimgPrueba1.jpg');

%%%% aplicando blur con "disk"
tic
diskFilter = fspecial('disk',10);
diskBlur = objectBlur(I,diskFilter,bI);
timedisk = toc
%%%% aplicando blur con "average"
tic
averageFilter = fspecial('average',[3 3]);
averageBlur = objectBlur(I,averageFilter,bI);
timeaverage = toc

%%% aplicando blur con "motion"
tic
motionFilter = fspecial('motion',10,270);
motionBlur = objectBlur(I,motionFilter,bI);
timemotion = toc

%%% aplicando blur con "gaussian"
tic
gaussianFilter = fspecial('gaussian',5,10);
gaussianBlur = objectBlur(I,gaussianFilter,bI);
timegaussian = toc

figure('name','Imagen original','numberTitle','off')
imshow(I)

figure('name','Imagen con disk blur','numberTitle','off')
imshow(diskBlur)
imwrite(diskBlur,'resultadosPruebas/diskBlurimgPrueba1.jpg');

figure('name','Imagen con motion blur','numberTitle','off')
imshow(motionBlur)
imwrite(motionBlur,'resultadosPruebas/motionBlurimgPrueba1.jpg');

figure('name','Imagen con gaussian','numberTitle','off')
imshow(gaussianBlur)
imwrite(gaussianBlur,'resultadosPruebas/gaussianBlurimgPrueba1.jpg');

figure('name','Imagen con average','numberTitle','off')
imshow(averageBlur)
imwrite(averageBlur,'resultadosPruebas/averageBlurimgPrueba1.jpg');

close all;
