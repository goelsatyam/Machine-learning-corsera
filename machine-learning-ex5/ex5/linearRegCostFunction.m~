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

%part1
X
y
theta
x=X*theta
x=x-y
l=theta.*theta
l=l(2:end,:)
l=l/m
l=l/2
l=l*lambda
l=sum(l)
x=sum(x.*x)/(2*m)+l
J=x
%part2
x=X*theta
x=x-y
x=X'*x
s=theta
s(1,1)=0;
s=s*lambda
x=x+s
x=x/m
grad=x
% =========================================================================

grad = grad(:);

end
