function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
[p,n] = size(X);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
sum = 0;

for i = 1:m
  h = sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) + theta(3)*X(i,3));  
  sum = sum - ( y(i) * log(h) +(1-y(i))*(log(1-h)));
end

tempsum = 0;
tempsum1 = 0;
tempsum2=0;

 for j = 1:m
    h = sigmoid(theta(1)*X(j,1) + theta(2)*X(j,2) + theta(3)*X(j,3));
    holder = (h-y(j));
    tempsum = tempsum + holder;
    tempsum1 = tempsum1 +  holder * X(j,2);
    tempsum2 = tempsum2 + holder * X(j,3);
 end 

grad(1) = tempsum/m;
grad(2) = tempsum1/m;
grad(3) = tempsum2/m;
J = sum/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
