%%
%% This program is based on Coursera Machine Learning ex4
%%

%% load package
pkg load strings;

%% Initialization
clear ; close all; clc

%% パラメーターを指定する
input_layer_size  = 2304; % 48x48のサイズの画像
hidden_layer_size = 802;  % 隠れ層のサイズ、だいぶ適当
kana_labels       = 73;   % ひらがなは濁音、半濁音含めて73ある

printf("========================================\n");
printf("=== 入力層、隠れ層、出力層を設定する ===\n");
printf("========================================\n");
printf("Input Layer size %d  \n", input_layer_size);
printf("Hidden Layer size %d \n", hidden_layer_size);
printf("Output Layer size %d \n", kana_labels);

fd = fopen('sample.csv', 'r');
raw = textscan(fd, '%s%s%s', 'delimiter', ',', 'headerLines', 1 );
fclose(fd);

X = cell2mat(raw);
m = rows(X);

printf("========================================\n");
printf("=== データセットの数を表示           ===\n");
printf("========================================\n");
printf("Dataset size m = %d \n", m);

printf("============================================\n");
printf("=== ニューラルネットワークの重みを初期化 ===\n");
printf("============================================\n");
fprintf('\nInitializing Neural Network Parameters ...\n')

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, kana_labels);

printf("Theta1 rows: %d, columns: %d\n", rows(Theta1), columns(Theta1));
printf("Theta2 rows: %d, columns: %d\n", rows(Theta2), columns(Theta2));

nn_params = [Theta1(:) ; Theta2(:)];

%% unicode    = cell2mat(raw{1}(1));
%% filename   = cell2mat(raw{2}(1));
%% img_base64 = cell2mat(raw{3}(1));

%% save & load temporary file
%% x  = base64decode(img_base64);
%% fd = fopen('buffer.png', 'wb');
%% fwrite(fd, x, 'uint8');
%% fclose(fd);

%% I = imread('buffer.png');

% Gray Image
%% colormap(gray);
%% imshow(I);
