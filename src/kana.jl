#
# This program is based on Coursera Machine Learning ex4
#

# load packages
# Pkg.add("Glob")
# Pkg.add("Color")
# Pkg.add("Images")
# Pkg.add("ImageMagick")
# Pkg.add("FileIO")

using FileIO
using Glob
using Images
using Colors

include("sigmoidGradient.jl");
include("randInitializeWeights.jl")
include("nnCostFunction.jl")
include("fmincg.jl")
include("predict.jl")

function main()

    println("This is test program")

    #
    # Specify parameters, training set
    #
    input_layer_size  = 48^2             # Images 48x48 pixel
    hidden_layer_size = 60               # Hidden layer size, I don't have any intension
    kana_labels       = 15               # 'Kana' has 73 characters, you need to reduce labels up to machine spec
    sample_size       = 30               # Take 30 samples for each characters
    test_set_size     = 10               # Take 1/3 samples for test set

    println("========================================\n")
    println("=== 入力層、隠れ層、出力層を設定する     ===\n")
    println("========================================\n")
    @printf("Input  Layer size %d \n", input_layer_size)
    @printf("Hidden Layer size %d \n", hidden_layer_size)
    @printf("Output Layer size %d \n", kana_labels)

    X = zeros(kana_labels * sample_size, input_layer_size)
    y = zeros(kana_labels * sample_size, 1)

    kanas = glob("./hiragana73/*/")
    for i = 1:kana_labels
        @printf("\nPick up from %s %d files.\n ", kanas[i], sample_size)
        pngs = glob("*.png", kanas[i])

        for j = 1:sample_size
            index = sample_size * (i-1) + j
            v     = load(pngs[j])
            v     = convert(Image{Gray}, v)
            # To deal with RGB using images, convert it to Gray scaled image
            # Create dataset as transposed X
            X[index,:] = v[:]
            y[index]   = i
            @printf(".")
        end
    end

    m = size(X,1)

    @printf("\n")
    println("========================================")
    println("=== データセットの数を表示           ===")
    println("========================================")
    @printf("Dataset size m = %d \n", m)

    println("============================================")
    println("=== ニューラルネットワークの重みを初期化 ===")
    println("============================================")
    @printf("\nInitializing Neural Network Parameters ...\n")

    Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    Theta2 = randInitializeWeights(hidden_layer_size, kana_labels)

    @printf("Theta1 rows: %d, columns: %d\n", size(Theta1, 1), size(Theta1, 2))
    @printf("Theta2 rows: %d, columns: %d\n", size(Theta2, 1), size(Theta2, 2))

    nn_params = [Theta1[:] ; Theta2[:]];

    println("============================================")
    println("=== 目的関数を求め、勾配を求める         ===")
    println("============================================")

    # Gradientを求める(OptimsetはただのHashMapに変更)
    it = 50
    options = Dict([("MaxIter", it)])

    # 正規化パラメーター
    # これでオーバーフィッティングを防ぐ
    lambda = 0.01;

    @printf("\n繰り返し回数: %d, 正規化パラメーター: %f\n", it, lambda)

    # 目的関数を作る関数を設定
    # nnCostFunctionは"p"だけが後続の処理で変数を受け取るクロージャー
    # 入力層とか隠れ層、そして出力層などの設定値はこの時点で確定
    costFunction = p -> nnCostFunction(p,
                                       input_layer_size,
                                       hidden_layer_size,
                                       kana_labels, X, y, lambda)

    # 最急降下法のアルゴリズムを使ってJ(θ)を最小化
    println("=============================================================")
    println("=== 最急降下法のアルゴリズムを使ってJ(θ)を最小化         ===")
    println("=============================================================")

    nn_params, cost = fmincg(costFunction, nn_params, options)

    # ニューラルネットワークの重みを取り出す
    Theta1 = reshape(nn_params[1:hidden_layer_size * (input_layer_size + 1)],
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):end],
                     kana_labels, (hidden_layer_size + 1));

    # 実際の画像を与えてみる
    pred = predict(Theta1, Theta2, X)

    #@printf("\n*** pred vs answer ***\n");
    #for i = 1:size(pred, 1)
    #    @printf("予測 %d, 正解 %d\n", pred[i], y[i])
    #end

    # トレーニングセットの正答率を出す
    # 10種類仮名を与えて正答率10%の場合ほぼまったく合っていないことになる
    @printf("\nトレーニングセットの正解率: %f ％\n", mean( countnz((pred .== y) .== true) / length(pred .== y) ) * 100)
    # テストセットのデータセットを用意する。これは訓練に使ったデータではないので
    # 機械学習がこれを正答できた場合、新たな類似画像でも対応できることを意味する
    Xv = zeros(kana_labels * test_set_size, input_layer_size)
    yv = zeros(kana_labels * test_set_size, 1)

    println(kana_labels * test_set_size)

    kanas = glob("./hiragana73/*/")
    index = 0

    for i = 1:kana_labels
        @printf("\nPick up from %s %d files.\n ", kanas[i], test_set_size)
        pngs = glob("*.png", kanas[i])

        for j = sample_size:(sample_size+test_set_size-1)
            index = index + 1
            v = load(pngs[j])
            v = convert(Image{Gray}, v)
            # To deal with RGB using images, convert it to Gray scaled image
            # Create dataset as transposed X
            Xv[index,:] = v[:]
            yv[index]   = i
            @printf(".")
        end
    end
    # 画像を与えてみる
    pred = predict(Theta1, Theta2, Xv)

    # テストセットの正答率を出す
    @printf("\nテストセットの正解率: %f ％\n", mean( countnz((pred .== yv) .== true) / length(pred .== yv) ) * 100)
end

main()
