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

function main()

    println("This is test program")

    # specify parameters
    input_layer_size  = 48^2 # Images 48x48 pixel
    hidden_layer_size = 60   # Hidden layer size, I don't have any intension
    kana_labels       = 10   # 'Kana' has 73 characters, you need to reduce labels up to machine spec
    sample_size       = 30   # Take 30 samples for each characters

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

            #@printf( "typeof(X[index,:]) = %s , typeof(v) = %s \n", typeof(X[index,:]), typeof(v))
            #@printf( "typeof(y[index]) = %s , typeof(v) = %s \n", typeof(y[index]), typeof(i))

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
    it = 200
    options = Dict([("MaxIter", it)])

    # 正規化パラメーター
    # これでオーバーフィッティングを防ぐ
    lambda = 0.01;

    @printf("\n繰り返し回数: %d, 正規化パラメーター: %f\n", it, lambda)

    # 目的関数を作る関数を設定
    costFunction = p -> nnCostFunction(p,
                                       input_layer_size,
                                       hidden_layer_size,
                                       kana_labels, X, y, lambda)

    # 最急降下法のアルゴリズムを使ってJ(θ)を最小化
    println("=============================================================")
    println("=== 最急降下法のアルゴリズムを使ってJ(θ)を最小化         ===")
    println("=============================================================")

    nn_params, cost = fmincg(costFunction, nn_params, options)
end

main()
