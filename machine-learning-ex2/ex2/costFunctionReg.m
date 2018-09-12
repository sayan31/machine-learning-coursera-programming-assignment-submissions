function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta);% This is used in the regularization calculation for gradients.

hvector = sigmoid(X*theta);%Hypothesis Vector

J =((1/m)*(sum( (-(y.*log(hvector))) - ((1-y).*log(1-hvector)) ))); %Original cost function value.

theta_regularized = theta(2:n); %Excluding the theta(0) parameter, which is theta(1) in Octave.
J+=(lambda/(2*m))*(theta_regularized'*theta_regularized); %Add the regularized value to the original cost function value.

grad = ((1/m)*(X'*(hvector-y))); %Reversing the (hvector-y) and X terms, and taking the transpose of the matrix X. This allows for correct dimensions of the vector product.
grad(2:n) += (lambda/m)*theta_regularized;





% =============================================================

end
