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

n = size(theta);
tempsum = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
sum = 0;
regParam = ((theta' * theta))*(lambda/(2*m));

for i=1:m
  h = sigmoid(X(i,:) * theta);
  sum = sum - ( y(i) * log(h) +(1-y(i))*(log(1-h)));  
  
  for j = 1:n
    holder = (h-y(i));
    tempsum(j) = tempsum(j)+holder*X(i,j);
  end  
end

tempsum = tempsum/m;


for j = 2:n
  tempsum(j) = tempsum(j) + (lambda/m) * theta(j);
end

J = (sum/m) + regParam;
grad = tempsum;

% =============================================================

end
