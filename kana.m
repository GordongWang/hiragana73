%%
%% This program is based on Coursera Machine Learning ex4
%%

%% Initialization
clear ; close all; clc

%% パラメーターを指定する
input_layer_size  = 2304; % 48x48のサイズの画像
hidden_layer_size = 802;  % 隠れ層のサイズ、だいぶ適当
kana_labels       = 73;   % ひらがなは濁音、半濁音含めて73ある

printf("Input Layer size %d \n", input_layer_size);
printf("Hidden Layer size %d \n", hidden_layer_size);
printf("Output Layer size %d \n", kana_labels);

fd = fopen('sample.csv', 'r');
raw = textscan(fd, '%s%s%s', 'delimiter', ',', 'headerLines', 1 );
fclose(fd);

unicode    = cell2mat(raw{1}(1));
filename   = cell2mat(raw{2}(1));
img_base64 = cell2mat(raw{3}(1));

%% save & load temporary file
pkg load strings;
x  = base64decode(img_base64);
fd = fopen('buffer.png', 'wb');
fwrite(fd, x, 'uint8');
fclose(fd);

I = imread('buffer.png');

% Gray Image
colormap(gray);
imshow(I);
