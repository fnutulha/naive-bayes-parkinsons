function [ probability ] = gaussian(x, mu, std)
%GAUSSIAN Summary of this function goes here
%   Detailed explanation goes here
term1 = (1/sqrt(2*pi*power(std,2)));
term2 = exp((-0.5)*power((x-mu)/std,2));
probability = term1*term2;
if probability > 1
%     pause
end
end


