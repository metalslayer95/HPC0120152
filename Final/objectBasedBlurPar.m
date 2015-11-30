I = gpuArray(imread('imgPrueba.jpg'));
nFilters = 3;

tic
sobelFilter = fspecial('sobel');
sobelImage= imfilter(I,sobelFilter,'replicate');
timesobel = toc;
% ibw = rgb2bw(I,
tic
blurFilter = fspecial('disk',5);
diskBlur = imfilter(I,blurFilter,'replicate');
timedisk = toc;
tic
motionFilter = fspecial('motion',20,45);
motionBlur = imfilter(I,motionFilter,'replicate','conv');
timemotion = toc;
tic
gaussianFilter = fspecial('gaussian',5,10);
gaussianBlur = imfilter(I,gaussianFilter,'replicate','conv');
timegaussian = toc;
subplot(1,nFilters+1,1)
imshow(I)
title('Original Image');

subplot(1,nFilters+1,2)
imshow(sobelImage)
title('Disk filtered image')

subplot(1,nFilters+1,3)
imshow(motionBlur)
title('Motion filtered image')


subplot(1,nFilters+1,4)
imshow(gaussianBlur)
title('Gaussian filtered image')

