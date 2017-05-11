# hiragana73

ひらがなの画像をOctaveで機械学習させたい

データセットは [文字画像データセット(平仮名73文字版)を試験公開しました](https://lab.ndl.go.jp/cms/hiragana73) から取得しています

## Just Run the sample

* Octaveパッケージのインストール
    * `strings`, `image` パッケージを入れます

```
# apt-get install liboctave-dev

$ octave --no-gui
>> pkg install -forge strings
>> pkg install -forge image
```


TBD...

## Install

* Dropboxに元のデータセットを用意しているのでダウンロードしてください

```
$ wget https://www.dropbox.com/s/jwt301cls9024l8/hiragana73.tar.gz?dl=0 -O hiragana73.tar.gz
$ tar xvf hiragana73.tar.gz
```

* CSVファイル作成

他の環境で使いやすいようにCSVファイルを作ります、８万枚データがあるので終わるまで気長に待ってください

```
$ bash ./preprocess.sh
```

これにより、画像データはBASE64で符号化され、以下のようなCSVファイルになります

```
unicode,filename,base64
U305E,1934_1235502_0067.png,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAABGdBTUEAALGPC...
```

手頃なサンプルデータがほしいので以下のようなコマンドを使ってsample.csvを作成しました

* これにより、５０音の全てに対して10個サンプルデータがあります

```
$ cat dataset.csv | awk -F ',' {'print $1'} | uniq | sort | xargs -I{} sh -c "grep '{}' dataset.csv | head" > sample.csv
```
