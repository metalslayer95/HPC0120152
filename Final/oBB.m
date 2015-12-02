I = imread('imgPrueba10.jpg');
auxI = I;
nFilters = 3;
[n,m,ch] = size(auxI);
Ibw = rgb2gray(I);
ind = find(Ibw < 130);
ind2 = find(Ibw >= 130);
Ibw(ind) = 0;
Ibw(ind2) = 255;
b = bwperim(Ibw,8); 
[B,L]= bwboundaries(b,'holes');  %Agujeros negros
figure(1)
fill=imfill(L,'holes');          %Llenar agujeros
Ibw = imfill(fill,'holes');
imshow(Ibw);


%%%% aplicando blur con "disk"
tic
diskFilter = fspecial('disk',10);
diskBlur = imfilter(I,diskFilter,'replicate');
for i=1:n 
     for j =1:m 
         if (Ibw(i,j) > 0)
             diskBlur(i,j,:) = auxI(i,j,:);
         end
     end
end
timedisk = toc
imshow(diskBlur)
return
%%% aplicando blur con "motion"
tic
motionFilter = fspecial('motion',10,270);
motionBlur = imfilter(I,motionFilter,'replicate','conv');
for i=1:n 
     for j =1:m 
         if (Ibw(i,j) > 0)
             motionBlur(i,j,:) = auxI(i,j,:);
         end
     end
end
timemotion = toc

%%% aplicando blur con "gaussian"
tic
gaussianFilter = fspecial('gaussian',5,10);
gaussianBlur = imfilter(I,gaussianFilter,'replicate','conv');
for i=1:n 
     for j =1:m 
         if (Ibw(i,j) > 0)
             gaussianBlur(i,j,:) = auxI(i,j,:);
         end
     end
end
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

% %%% graficando imagen con motion
% subplot(1,nFilters,2)
% imshow(motionBlur)
% title('Motion')
% 
%%% graficando imagen con gaussian
% subplot(1,nFilters,3)
% imshow(gaussianBlur)
% title('Gaussian')
