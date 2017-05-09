%% Initialization
clear ; close all; clc

printf("This is test program\n");

fd = fopen('sample.csv', 'r');
raw = textscan(fd, '%s%s%s', 'delimiter', ',', 'headerLines', 1 );
fclose(fd);

unicode    = raw{1}(1);
filename   = raw{2}(1);
img_base64 = raw{3}(1);

% save & load temporary file
% x  = base64_decode(img_base64);
% fd = fopen('buffer.png', 'wb');
% fwrite(fd, x, 'uint8');
% fclose(fd);
%
% I = imread('buffer.png');

% Gray Image
colormap(gray);

%imshow(I);
