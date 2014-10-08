function [X, Y] = LoadData()
%%LOADDATA 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 19 Sep 2014

%   Time-stamp: <2014-10-08 10:24:37 by xenuts>

%% MNIST
    load('dataset/mnist-train-N200.mat');
%%    load('dataset/mnist-train.mat');
   
    X = double(reshape(X, 28, 28, 1, 2000)) / 255.0;
    Y = double(c');

%% Track-Car4-0001
% % % load('../../Trials/VecTracking/Train900-Car4-0001.mat');
% % % X = double(X) / 255.0;
end