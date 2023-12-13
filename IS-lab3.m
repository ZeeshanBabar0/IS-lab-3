% Clear workspace and close figures
clear all;
close all;
clc
% Data preparation
x_values = linspace(0, 1, 23);
desired_output = ((1 + 0.6*sin(2*pi*x_values/0.7)) + (0.3*sin(2*pi*x_values))) / 2;

% Plot the original data plot 2
figure('Name', 'Function Approximation plot 2', 'NumberTitle', 'off');
plot(x_values, desired_output, 'k', 'LineWidth', 1.5);
grid on;


% Network parameters
center1 = 0.20;
center2 = 0.91;
radius1 = 0.15;
radius2 = 0.15;
learning_rate = 0.1;

% Initialize weights and bias
weight1 = rand() * 0.1;
weight2 = rand() * 0.1;
bias = rand() * 0.1;

% Training the neural network
epochs = 1000;
for epoch = 1:epochs
    for i = 1:length(x_values)
        % Calculate the output of the network
        activation1 = exp(-(((x_values(i)-center1)^2)/(2*radius1^2)));
        activation2 = exp(-(((x_values(i)-center2)^2)/(2*radius2^2)));
        output = activation1 * weight1 + activation2 * weight2 + bias;

        % Update weights and bias using the delta rule
        error = desired_output(i) - output;
        weight1 = weight1 + learning_rate * error * activation1;
        weight2 = weight2 + learning_rate * error * activation2;
        bias = bias + learning_rate * error;
    end
end

% Plot the learned function
learned_output = zeros(1, length(x_values));
for index = 1:length(x_values)
    activation1 = exp(-(((x_values(index)-center1)^2)/(2*radius1^2)));
    activation2 = exp(-(((x_values(index)-center2)^2)/(2*radius2^2)));
    learned_output(index) = activation1 * weight1 + activation2 * weight2 + bias;
end

% Plot the learned function plot 1
figure('Name', 'Learned Function plot 1', 'NumberTitle', 'off');
plot(x_values, learned_output, 'rx', 'LineWidth', 1.5);
grid on;

