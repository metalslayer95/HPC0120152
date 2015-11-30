I = imread('imgPrueba.jpg');
nFilters = 3;
%%%
[b, num]=CapBinaria(I);         %Obtener imagen en FORMATO BINARIO
[B,L]= bwboundaries(b,'holes');  %Agujeros negros
figure(1)
fill=imfill(L,'holes');          %Lenar agujeros
Ibw = imfill(fill,'holes');
[Ilabel, Ne]= bwlabel(Ibw);          %Ne numero de obejtos blancos
stat = regionprops(Ilabel, 'centroid');     %Obtener(Area,Centroide,Limites) 
imshow(I) 
hold on
for x = 1: numel(stat)
    if numel(stat(x))<2  
        plot(stat(x).Centroid(1),stat(x).Centroid(2),'x');   
        xc=stat(x).Centroid(1);
        yc=stat(x).Centroid(2);
        radius=25;
        theta = 0:0.01:2*pi;
        Xfit = radius*cos(theta) + xc;
        Yfit = radius*sin(theta) + yc;
        plot(Xfit, Yfit, 'y', 'LineWidth', 4);
        fprintf('\n--------OBJETO DETECTADO--------\n');
        fprintf('\nPosicion:\n');
        fprintf('Abscisa%10.3f\n',xc)
        fprintf('Ordenada%10.3f\n',yc)
    end
end
%%%%
figure(2)
%%% Filtrado con sobel
tic
sobelFilter = fspecial('sobel');
mask = padarray(2,[2 2]) - fspecial('gaussian' ,[5 5],2); % create unsharp mask
sharpImage= imfilter(I,mask,'replicate');
sobelImage = imfilter(sharpImage,sobelFilter,'replicate');
timesobel = toc;

%%%% aplicando blur con "disk"
tic
blurFilter = fspecial('disk',5);
diskBlur = imfilter(I,blurFilter,'replicate');
timedisk = toc;

%%% aplicando blur con "motion"
tic
motionFilter = fspecial('motion',20,45);
motionBlur = imfilter(I,motionFilter,'replicate','conv');
timemotion = toc;

%%% aplicando blur con "gaussian"
tic
gaussianFilter = fspecial('gaussian',5,10);
gaussianBlur = imfilter(I,gaussianFilter,'replicate','conv');
timegaussian = toc;

%%% graficando imagen original
subplot(1,nFilters+1,1)
imshow(I)
title('Original');

%%% graficando imagen con disk
subplot(1,nFilters+1,2)
imshow(sobelImage)
title('Disk')

%%% graficando imagen con motion
subplot(1,nFilters+1,3)
imshow(motionBlur)
title('Motion')

%%% graficando imagen con gaussian
subplot(1,nFilters+1,4)
imshow(gaussianBlur)
title('Gaussian')

%%%%
figure(3)
Ibw = rgb2gray(I);
ind = find(Ibw < 185);
ind2 = find(Ibw >= 185);
Ibw(ind) = 0;
Ibw(ind2) = 255;
I2 = bwperim(Ibw,8);
imshow(Ibw);
%hold on
%plot(I2);



