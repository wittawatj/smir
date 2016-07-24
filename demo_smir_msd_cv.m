% 
% 
% Demonstrate how to train a semi-supervised learning classifier with SMIR.
% 

% Set seed to make sure everything is deterministic.
RandStream.setGlobalStream(RandStream.create('mt19937ar','seed', 1));          

% load 2moons data. 
load a_2moons_1;

% Inside, X is the input matrix 2x556 (dxn).
% Y is 1x556 containing full labels. Y0 is 1x6 containing labels for only
% the first 6 points in X (3 for each class). 
% We will use Y0 for training. 
hold on 
% Here is the plot of the training data.
plot2dlabels(X, Y0);
title('Training data');
hold off

% Array of Gaussian width candidates for cross validation.
med = meddistance(X);
o.gaussianwidthlist = med*[1/3, 1/2];
    
% array of gamma's to try
o.gammalist = 10.^[-2, -1];

% Array of lambdafactors. Each of this value v will be used to calculate
% lambda = \gamma*c/n + v (c is the number of classes, n is the sample size).
% This is to ensure the convexity of the problem.
o.lambdafactorlist = 10.^[-5, -3];

% Number of folds in cross validation used to select gamma and Gaussian
% width
o.fold = 2;

% Perform cross validation (In this case, CV is performed using only 6 labeled points 
% in 2moons data). 
SMG  = smir_cv( X, Y0, o);

% SMG is a struct containing information obtained after doing cross
% validation. Here is an example of what it looks like inside SMG.
% 
%                    A: [556x2 double]
%                  Beta: [556x2 double]
%               Options: [1x1 struct]
%             bestgamma: 1.0000e-03
%     bestgaussianwidth: 0.4803
%      bestlambdafactor: 0.0100
%              mincverr: 0.5000
%                     q: [function_handle]
%               timecpu: 0.3900
%            timetictoc: 0.1864
% 
% In particular the most important is q, a function handle which can be
% applied to new test (unseen) points in a matrix of the form d x n'
SMG

% Get the model trained with the best parameters chosen by the cross
% validation.
condProbFunc = SMG.q;

% Note that in general the learned model of SMIR can be applied to
% unseen points. But here, for demonstration purpose, we test it on the 
% matrix X.
Yprob = condProbFunc(X);

% The result Yprob will be a c x n' matrix. Yprob(i,j) indicates the
% probability that example j belongs to class i.
% 
% Here we convert the result to the most probable classes.
[Temp, Yh] = max(Yprob, [] , 1); % Yh is now 1 x n' whose entries are in {1,..,c}

% Plot the labeled results 
figure; 
hold on 
plot2dlabels(X, Yh);
title('Test results');
hold off

% Make an entropy plot to see which region the model is
% confident in predicting. Blue denotes low entropy region (high
% confident).
plot2dentropy(X, Y0, condProbFunc);

% In this plot, the red S-shape region path is running in between
% the two moons. It has high entropy since the region lies between the two
% classes. Entropy is relatively low outside this region.

% Also try 
plot2dclassprob(X,Y0, condProbFunc);
% to see the decision boundary 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
