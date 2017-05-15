#
# This program is based on Coursera Machine Learning ex4
#

# load packages
using FileIO
using Images
using Glob

include("sigmoidGradient.jl");

function main()

    println("This is test program")

    # specify parameters
    input_layer_size  = 48^2 # Images 48x48 pixel
    hidden_layer_size = 60   # Hidden layer size, I don't have any intension
    kana_labels       = 10   # 'Kana' has 73 characters, you need to reduce labels up to machine spec
    sample_size       = 100  # Take 30 samples for each characters

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
            v     = (load(pngs[j])[:])'

            #@printf( "typeof(X[index,:]) = %s , typeof(v) = %s \n", typeof(X[index,:]), typeof(v))
            #@printf( "typeof(y[index]) = %s , typeof(v) = %s \n", typeof(y[index]), typeof(i))

            # to deal with RGB using images, convert it to Gray scaled image
            X[index,:] = convert(Image{Gray}, v)
            y[index]   = i
            @printf(".")
        end
    end

    m = size(X,1)

    @printf("\n")
    println("========================================");
    println("=== データセットの数を表示              ===");
    println("========================================");
    @printf("Dataset size m = %d \n", m);

end

main()
