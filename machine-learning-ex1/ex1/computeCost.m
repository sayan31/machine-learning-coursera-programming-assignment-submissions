function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


hvector = X*theta; %This creates the hypothesis vector used to plug into the formula for calculating the cost function.
J+=((1/(2*m))*(sum(((hvector-y).^2)))); %increments J by the computed cost value.



% =========================================================================

end
