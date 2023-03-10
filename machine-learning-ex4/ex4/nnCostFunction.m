function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];
first_hidden_layer = sigmoid(X * Theta1');
first_hidden_layer = [ones(m,1) first_hidden_layer];
output_layer = sigmoid(first_hidden_layer * Theta2');

for i = 1:m
    temp_x = output_layer(i,:);
    
    temp_y = zeros(num_labels,1);
    temp_y(y(i)) = 1;
    
    first_exp = log(temp_x) * temp_y;
    second_exp = log(1-temp_x) * (1-temp_y);
    J = J + first_exp + second_exp;
end

J = J * (-1) * (1/m);

temp_theta1 = Theta1;
temp_theta2 = Theta2;
temp_theta1(:,1) = 0;
temp_theta1 = temp_theta1.^2;
temp_theta2(:,1) = 0;
temp_theta2 = temp_theta2.^2;
regularized = lambda * (1/(2*m)) * (sum(temp_theta1(:)) + sum(temp_theta2(:)));

J = J + regularized;



for t = 1:m
    layer1 = X(t,:);                % 1x401
    layer2 = sigmoid(layer1 * Theta1');   % 1x401 * 401x25 = 1x25 
    temp_layer2 = [1 layer2];             % 1x26
    layer3 = sigmoid(temp_layer2 * Theta2'); % 1x26 * 26x10 = 1x10
    
    temp_y = zeros(num_labels, 1); % 10x1
    temp_y(y(t)) = 1;
    
    delta_layer_3 = layer3 - temp_y'; % 1x10
    
    temp_delta = delta_layer_3 * Theta2; % 1x10 * 10x26 = 1x26
    delta_layer_2 = temp_delta(2:end) .* sigmoidGradient(layer1 * Theta1');
    % 1x25 .* 1x25 = 1x25
    
    Theta1_grad = Theta1_grad + (delta_layer_2' * layer1); % 25x401
    Theta2_grad = Theta2_grad + (delta_layer_3' * temp_layer2); % 10x26
    
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

temp_theta1 = Theta1;
temp_theta2 = Theta2;
temp_theta1(:,1) = 0;
temp_theta2(:,1) = 0;
regularizaion_theta1 = (lambda/m) * temp_theta1;
regularizaion_theta2 = (lambda/m) * temp_theta2;

Theta1_grad = Theta1_grad + regularizaion_theta1;
Theta2_grad = Theta2_grad + regularizaion_theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
