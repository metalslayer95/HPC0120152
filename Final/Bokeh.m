function [output] = Bokeh(input, radius)

% Function that performs blurring on the whole image except a user defined
% ROI,using a disk kernel. The effect resembles the bokeh effect.
% Inputs:
%        input  - Input image
%        radius ? User's choice of radius for the disk kernel
% Output:
%        output - Output image (only user-defined ROI stays in focus)

kernel = fspecial('disk',radius);       % Create disk kernel
disp('Select area to keep in focus!')   % Display message to user
mask = roipoly(input);           % User selects area of interest
output = [];                     % Start with an empty image
for i = 1:size(input,3)          % Covering the case of color images
    cropped = input(:,:,i);      % Perform per-channel processing
    channel = input(:,:,i);             % Replica of channel
    cropped(mask == 1) = 0;             % Keep only ROI outside mask
    cropped = imfilter(cropped,kernel); % Perform blurring out of ROI
    channel(mask==0) = cropped(mask==0); % Only keep ROI unaffected
    output = cat(3,output,channel);  % Concatenate channels
end
figure(1)
imshow(input)
figure(2)
imshow(output)