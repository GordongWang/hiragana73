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

Windows10, Debian GNU/Linuxでテストしています

* ニューラルネットワークの起動 (Julia)
    * `Pkg.clone` で依存関係を解決できます

```
$ julia -e 'Pkg.clone("https://github.com/Hiroyuki-Nagata/hiragana73.git")'
```

* `Pkg.dir("***")` でインストール場所を調べられます

```
$ julia
julia> Pkg.dir("hiragana73")
"~/.julia/v0.4/hiragana73"
```

* 移動してデータセットをダウンロード

```
$ cd ~/.julia/v0.4/hiragana73/src
$ wget https://www.dropbox.com/s/jwt301cls9024l8/hiragana73.tar.gz?dl=0 -O hiragana73.tar.gz
$ tar xvf hiragana73.tar.gz
```

* `include("kana.jl")` で起動

```
julia> cd("~/.julia/v0.4/hiragana73/src")
julia> include("kana.jl")
```

## メモ

* 判別対象のターゲットの数は `kana_labels` で指定出来るようにしている
    * `kana_labels = 10` だと `あ,い,う,え,お,か,が,き,ぎ,く` までの画像を判別するように機械学習する
    * 家庭用のパソコン（クロック数3.8GHz）でテストしている

### Julia

今のところ下記のパラメーターだとうまく動いた。それ以上は計算に時間がかかりすぎる。

とりあえずテストセットにおいても80%以上の正解率はいくはず

```
input_layer_size  = 48^2  ## 48x48のサイズの画像
hidden_layer_size = 60    ## 隠れ層のサイズ、だいぶ適当
kana_labels       = 15    ## ひらがなは濁音、半濁音含めて73ある、収束しない場合は少なくする
sample_size       = 50    ## 標本数を100とる（トレーニングセット）
test_set_size     = 15    ## 機械学習が学習した後の新規テスト用画像の枚数
```

### Octave

今のところ下記のパラメーターだとうまく動いた。それ以上は計算に時間がかかりすぎる。

```
input_layer_size  = 48^2; %% 48x48のサイズの画像
hidden_layer_size = 60;   %% 隠れ層のサイズ、だいぶ適当
kana_labels       = 10;   %% ひらがなは濁音、半濁音含めて73ある、収束しない場合は少なくする
sample_size       = 100;  %% それぞれの標本数を100とる
```

判別対象とネットワークの階層をもっと増やしたいがPCのスペックが足りない。やっぱりGPUが必要だ！
