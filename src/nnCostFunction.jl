include("sigmoidGradient.jl")

function nnCostFunction(nn_params,
                        input_layer_size,
                        hidden_layer_size,
                        data_with_labels,
                        X, y, lambda)
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, data_with_labels, ...
    #    X, y, lambda) computes the cost and gradient of the neural network. The
    #    parameters for the neural network are "unrolled" into the vector
    #    nn_params and need to be converted back into the weight matrices.
    #
    #    The returned parameter grad should be a "unrolled" vector of the
    #    partial derivatives of the neural network.
    #

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    #
    # reshape (a,m,n)
    #
    # reshape ([1, 2, 3, 4], 2, 2)
    # ->  1  3
    #     2  4
    #
    Theta1 = reshape(nn_params[1:hidden_layer_size * (input_layer_size + 1)]
                     , hidden_layer_size, (input_layer_size + 1))
    Theta2 = reshape(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):end]
                     ,data_with_labels, (hidden_layer_size + 1))

    #@printf("Theta1 x: %d, y: %d\n", size(Theta1, 1), size(Theta1, 2))
    #@printf("Theta2 x: %d, y: %d\n", size(Theta2, 1), size(Theta2, 2))

    m = size(X, 1)

    J = 0
    Theta1_grad = zeros(size(Theta1))
    Theta2_grad = zeros(size(Theta2))

    y_alt = eye(data_with_labels)

    D1 = zeros(size(Theta1))
    D2 = zeros(size(Theta2))

    # Backpropagation Updates
    for t = 1:m
        # 1, 入力層 a1 にデータセットを入れる
        #@printf("Datasets: Rows: %d, Cols: %d \n", size(X[t,:],1), size(X[t,:],2))

        act1 = [ones(1,1); X[t,:]]
        z2   = Theta1 * act1
        act2 = [ones(1,1); sigmoid(z2)]
        z3   = Theta2 * act2
        # see: http://stackoverflow.com/questions/29159386/how-should-i-convert-a-singleton-array-to-a-scalar
        # Converting a singleton matrix as a scalar
        act3 = reshape([sigmoid(z3)], 1)[1]

        #@printf("Act1: Rows: %d, Cols: %d \n", size(act1,1), size(act1,2))
        #@printf("Act2: Rows: %d, Cols: %d \n", size(act2,1), size(act2,2))
        #@printf("Act3: Rows: %d, Cols: %d \n", size(act3,1), size(act3,2))

        # 目的関数
        #println(- y_alt[:, Int64(y[t])]')
        #println(log(act3))
        #println(- (1 - y_alt[:, Int64(y[t])])')
        #println(log(1-act3))

        J += - y_alt[:, Int64(y[t])]' * log(act3) - (1 - y_alt[:, Int64(y[t])])' * log(1-act3)
        # 2, レイヤー３の出力Kのそれぞれに対して
        # 3, δ2を計算
        delta_3 = act3 - y_alt[:, Int64(y[t])]
        delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2])
        delta_2 = delta_2[2:size(delta_2, 1),:]

        # 4
        # D1 = D1 + d2 * a1'
        #@printf("Act1(T): Rows: %d, Cols: %d \n", size(act1',1), size(act1',2))
        #@printf("Act2(T): Rows: %d, Cols: %d \n", size(act2',1), size(act2',2))
        #@printf("d2     : Rows: %d, Cols: %d \n", size(delta_2,1), size(delta_2,2))
        #@printf("d3     : Rows: %d, Cols: %d \n", size(delta_3,1), size(delta_3,2))
        D1 += (delta_2 * act1')
        D2 += (delta_3 * act2')
    end

    # 目的関数の収束
    J = J / m

    p1 = Theta1[:, 2:size(Theta1,2)].^2
    p2 = Theta2[:, 2:size(Theta2,2)].^2
    r = sum(sum(p1)) + sum(sum(p2))
    r = r / (2 * m)
    J = J + lambda * r

    Theta1_grad = D1 / m + lambda * Theta1
    Theta2_grad = D2 / m + lambda * Theta2

    grad = [Theta1_grad[:] ; Theta2_grad[:]]

    # multiple values can be returned from a function using tuples
    # if the return keyword is omitted, the last term is returned
    return J, grad
end
