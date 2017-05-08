%% Initialization
clear ; close all; clc

printf("This is test program\n");

dirs = glob('./hiragana73/*');
for i=1:numel(dirs)
  printf("index[%02d] = %s \n", i, dirs{i});
endfor

% Gray Image
colormap(gray);

imshow("./hiragana73/U3042/1900_753325_0060.png");
