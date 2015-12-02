function [output] = objectBlur(input,kernel,mask)
output = [];
for i = 1:size(input,3)          % Covering the case of color images
    cropped = input(:,:,i);      % Perform per-channel processing
    channel = input(:,:,i);             % Replica of channel
    cropped(mask == 1) = 0;             % corta el objeto de la imagen
    cropped = imfilter(cropped,kernel); % aplica el kernel a la imagen sin el objeto
    channel(mask==0) = cropped(mask==0); % Only keep ROI unaffected
%  imshow(channel)
%  pause(2);
    output = cat(3,output,channel);  % Concatenate channels
end