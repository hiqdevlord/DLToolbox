%% Convolutional Network Layer Processing Routines

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Image Preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Normalization
%  - Within / across feature maps
%  - Before / after pooling
%  [X] Local Contrast Normlization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Layer Processing Routines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Convolution / Filter Bank (Learnable)
%  + Why?
%    1. Dependencies are local
%    2. Translation incarance
%    3. Few parameters (filter weights)
%    4. Stride can be greater than 1 (faster, less memory)
%  [ ] GPU Acceleration

%% Non-Linearity
%  [ ] Sigmoid
%  [ ] tanh
%  [X] Rectified Linear Unit (ReLU)
%      - Simplifies backpropagation
%      - Makes learning faster
%      - Avoid saturation issues
%  [ ] Component-wise Shrinkage
%  [ ] Winner-Takes-All

%% Pooling and Subsampling
%  + Why?
%    - Invariance to small transformations
%    - Larger receptive fields (see more of input)
%  [X] Max Pooling
%  [ ] Average Pooling
%  [ ] Sum Pooling

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Learning in Practice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - Use cross-entropy loss for classification
% - Use Stochastic Gradient Descent on minibatches
% - Shuffle the training samples
% - Schedule to decrease the learning rate
% - Use a bit of L1 or L2 regularization on the weights (or a combination)
%   But it's best to turn it on after a couple of epochs
% - Use "dropout" for regularization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% !!! lots more in "Neural Networks, Tricks of the Trade" (2012 edition)