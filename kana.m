%%
%% This program is based on Coursera Machine Learning ex4
%%

%% load package
pkg load strings;

%% Initialization
clear ; close all; clc
page_screen_output(0);
page_output_immediately(1);

%% パラメーターを指定する
input_layer_size  = 48^2; %% 48x48のサイズの画像
hidden_layer_size = 200;  %% 隠れ層のサイズ、だいぶ適当
kana_labels       = 73;   %% ひらがなは濁音、半濁音含めて73ある
sample_size       = 10;   %% それぞれの標本数を30とる

printf("========================================\n");
printf("=== 入力層、隠れ層、出力層を設定する ===\n");
printf("========================================\n");
printf("Input Layer size %d  \n", input_layer_size);
printf("Hidden Layer size %d \n", hidden_layer_size);
printf("Output Layer size %d \n", kana_labels);

X = zeros(kana_labels * sample_size, input_layer_size);
y = zeros(kana_labels * sample_size, 1);

kanas = glob("./hiragana73/*/");

for i = 1:rows(kanas),
  printf("\nPick up from %s %d files.\n ", kanas{i}, sample_size);
  pngs = glob([kanas{i} "*.png"]);


  for j = 1:sample_size,
    index = sample_size * (i-1) + j;
    vec   = imread(pngs{j})(:)';

    if (columns(vec) > input_layer_size)
      printf("index: %d, skipping...\n", index);
      vec = imread(pngs{j+30})(:)';
      if (columns(vec) > input_layer_size)
	error("still large size image exists...\n");
      endif
    else
      printf(".");
    endif
    %%printf("index: %d, vector row: %d, col: %d\n", index, rows(vec), columns(vec));

    X(index, :) = vec; %% データセットをベクタで与える
    y(index)	= i;   %% 教師あり学習の答えを設定しておく
  endfor
endfor

m = rows(X);


printf("\n");
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

printf("============================================\n");
printf("=== 目的関数を求め、勾配を求める         ===\n");
printf("============================================\n");

%% Octaveの機能でGradientを求める
options = optimset('MaxIter', 3);

%% 正規化パラメーター
%% これでオーバーフィッティングを防ぐ
lambda = 1;

%% 目的関数を作る関数を設定
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   kana_labels, X, y, lambda);

%% 最急降下法のアルゴリズムを使ってJ(θ)を最小化
[nn_params, cost] = fmincg(costFunction, nn_params, options);

printf("Theta: %d Cost: %d \n", nn_params, cost);

% Gray Image
colormap(gray);
