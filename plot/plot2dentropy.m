function plot2dentropy(X, Yc, q, options)
%
% Plot entropy of class conditional probability. X must be 2D.
% q is a function handle to the estimated conditional probability
% function.
%
if size(X,1) ~= 2
    error('%s only works for 2D data.', mfilename);
end

if nargin < 4 || isempty(options) 
    options =[] ;
end

% plot
steps = 120;

x = X(1,:);
y = X(2,:);
xlist = linspace(min(x), max(x), steps);
ylist = linspace(min(y), max(y), steps);
[xx yy] = meshgrid(xlist, ylist);
regionYh = q([xx(:)';yy(:)']);

% calculate entropy
E = -nansum(regionYh.*log2(regionYh), 1);
ee = reshape(E, length(ylist), length(xlist));

% Normalize entropy to make it 0 to 1
c = length(unique(Yc));
uniform_ent = log2(c);
ee = ee/uniform_ent;

% reverse Y to match with imagesc usage
ee = ee(length(ylist):-1:1, :);

figure
imagesc(ee);
colorbar
daspect([1 1  1])

% xlabel('x1');
% ylabel('x2');
% title(sprintf('Entropy based on conditional probability estimated by %s', ...
%     func2str(SM.Options.smitfunc) ));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end