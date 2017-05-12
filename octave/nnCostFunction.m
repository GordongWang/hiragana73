function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   data_with_labels, ...
                                   X, y, lambda)
  %% NNCOSTFUNCTION Implements the neural network cost function for a two layer
  %% neural network which performs classification
  %%    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, data_with_labels, ...
  %%    X, y, lambda) computes the cost and gradient of the neural network. The
  %%    parameters for the neural network are "unrolled" into the vector
  %%    nn_params and need to be converted back into the weight matrices.
  %%
  %%    The returned parameter grad should be a "unrolled" vector of the
  %%    partial derivatives of the neural network.
  %%

  %% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  %% for our 2 layer neural network
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   data_with_labels, (hidden_layer_size + 1));

  m = size(X, 1);

  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  y_alt = eye(data_with_labels);

  D1 = zeros(rows(Theta1), columns(Theta1));
  D2 = zeros(rows(Theta2), columns(Theta2));

  %% Backpropagation Updates
  for t = 1:m,
    %% 1, 入力層 a1 にデータセットを入れる
    act1 = [ones(1,1); X(t,:)'];
    z2   = Theta1 * act1;
    act2 = [ones(1,1); sigmoid(z2)];
    z3   = Theta2 * act2;
    act3 = [sigmoid(z3)];

    %% 目的関数
    J += - y_alt(:, y(t))' * log(act3) - (1 - y_alt(:, y(t)))' * log(1-act3);
    %% 2, レイヤー３の出力Kのそれぞれに対して
    %% 3, δ2を計算
    delta_3 = act3 - y_alt(:, y(t));
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2]);
    delta_2 = delta_2(2:end, :);

    %% 4
    %% D1 = D1 + d2 * a1'
    D1 += (delta_2 * act1');
    D2 += (delta_3 * act2');
  end

  %% 目的関数の収束
  J = J / m;

  p1 = Theta1(:, 2:end).^2;
  p2 = Theta2(:, 2:end).^2;
  r = sum(sum(p1)) + sum(sum(p2));
  r = r / (2 * m);
  J = J + lambda * r;

  Theta1_grad = D1 / m + lambda * Theta1;
  Theta2_grad = D2 / m + lambda * Theta2;

  grad = [Theta1_grad(:) ; Theta2_grad(:)];
  %% printf("J = %d\t, grad = %d \n", J, grad);
end
