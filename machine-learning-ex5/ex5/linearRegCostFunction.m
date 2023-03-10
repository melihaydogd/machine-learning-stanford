function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


predicted = X * theta;
subtraction = predicted-y;
summation_first = sum(subtraction.^2);
cost = (1/(2*m))*summation_first;

temp_theta = theta;
temp_theta(1) = 0;
summation_second = sum(temp_theta.^2);
regularization = (lambda/(2*m))*summation_second;

J = cost + regularization;

if size(X,1) == 1
    grad = (1/m)*(subtraction.*X);
else 
    grad = (1/m)*sum(subtraction.*X);
end
grad(:,2:end) = grad(:,2:end) + (lambda/m) * theta(2:end,:)';



% =========================================================================

grad = grad(:);

end
