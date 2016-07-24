function plot2dclassprob(X, Yc, q)
%
% Plot estimated class conditional probability for each class with
% color region. X must be 2D.
%
c =length(unique(Yc)); 
if c > 5
    error('Current %s supports up to 5 classes.', mfilename);
end

if size(X,1) ~= 2
    error('%s only works for 2D data.', mfilename);
end

% SM = smitfunc(X, Yc, options);
% q = SM.q; % estimated conditional prob. function

% Plot
steps = 180;

x = X(1,:);
y = X(2,:);
xlist = linspace(min(x), max(x), steps);
ylist = linspace(min(y), max(y), steps);
[xx yy] = meshgrid(xlist, ylist);
regionYh = q([xx(:)';yy(:)']);
[V eYc] = max(regionYh, [], 1);

figure
hold on

% Plot probability regions
zz = reshape(eYc, length(ylist), length(xlist));
contourf(xx, yy, zz);

xlabel('x1');
ylabel('x2');
title(sprintf('Conditional probability regions'));

% Define new colormap 
% 'r','b','m','y','c'
lightenfactor = 0.8;
CM = [1 0 0; 0 0 1; 1 0 1; 1 1 0; 0 1 1];
CM(CM == 0) = lightenfactor;
colormap(CM(1:c,:));
% cb = colorbar();
% set(cb,'YMinorTick', 'off');

% Plot 2D points 
plotstyles
Uy  = unique(Yc);
L = cell(1,length(Uy));
for yi=1:length(Uy)
    y = Uy(yi);
    Ind = Yc==y;
    plot(X(1,Ind), X(2,Ind), Markers(yi), 'MarkerSize', 9, ...
    'MarkerFaceColor', MarkerEdgeColors(yi), ...
    'MarkerEdgeColor', MarkerEdgeColors(yi) )
    L{yi} = sprintf('class %d', y);
end

if length(Yc) < size(X,2)
    % there are some unlabeled points
    plot(X(1,(length(Yc)+1):end), X(2,(length(Yc)+1):end), 'xk');
    L = [L {'unlabeled'} ];

end

legend([{'Probability contour'} L]);


grid on
box on
daspect([1 1  1])

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end