# hiragana73 [![Build Status](https://travis-ci.org/Hiroyuki-Nagata/hiragana73.svg?branch=master)](https://travis-ci.org/Hiroyuki-Nagata/hiragana73)

ひらがなの画像を Octave/Julia で機械学習させたい

データセットは [文字画像データセット(平仮名73文字版)を試験公開しました](https://lab.ndl.go.jp/cms/hiragana73) から取得しています

## Just Run the sample

### Octave

* 今のところ、Debian Linuxでしかテストしてません
    * Windowsでも動く気はしますが、それなりにスペックが必要そう

```
$ git clone https://github.com/Hiroyuki-Nagata/hiragana73.git
$ cd hiragana73/octave/
$ wget https://www.dropbox.com/s/jwt301cls9024l8/hiragana73.tar.gz?dl=0 -O hiragana73.tar.gz
$ tar xvf hiragana73.tar.gz
```

* ニューラルネットワークの起動 (Octave)

```
$ octave --no-gui
>> kana
```

### Julia

* Windows10, Debian GNU/Linuxでテストしています

```
$ git clone https://github.com/Hiroyuki-Nagata/hiragana73.git
$ cd hiragana73/src/
$ wget https://www.dropbox.com/s/jwt301cls9024l8/hiragana73.tar.gz?dl=0 -O hiragana73.tar.gz
$ tar xvf hiragana73.tar.gz
```

* ニューラルネットワークの起動 (Julia)
    * JuliaのREPLを起動してそれぞれのライブラリを入れる
	* `Pkg.add()` してうまくいかないときは、一度コンソールを抜けてやり直す

```
$ julia
julia> Pkg.update()

julia> Pkg.add("FileIO")
julia> Pkg.add("Glob")
julia> Pkg.add("Color")
julia> Pkg.add("Images")
julia> Pkg.add("ImageMagick")

julia> include("kana.jl")
```

## メモ

* 判別対象のターゲットの数は `kana_labels` で指定出来るようにしている
    * `kana_labels = 10` だと `あ,い,う,え,お,か,が,き,ぎ,く` までの画像を判別するように機械学習する

今のところ、下記のパラメーターだと家庭用のパソコン（クロック数3.8GHz）でもうまく動いた

```
input_layer_size  = 48^2; %% 48x48のサイズの画像
hidden_layer_size = 60;   %% 隠れ層のサイズ、だいぶ適当
kana_labels       = 10;   %% ひらがなは濁音、半濁音含めて73ある、収束しない場合は少なくする
sample_size       = 100;  %% それぞれの標本数を100とる
```

判別対象とネットワークの階層をもっと増やしたいがPCのスペックが足りない。やっぱりGPUが必要だ！
