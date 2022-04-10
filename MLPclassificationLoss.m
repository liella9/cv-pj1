function [f,g,gb] = MLPclassificationLoss(w,b,X,y,nHidden,nLabels)

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);
% bias

inputBias = b(1:nHidden(1));
offset = nHidden(1);
for h = 2:length(nHidden)
    hiddenBias{h-1} = b(offset+1:offset+nHidden(h));
    offset = offset + nHidden(h);
end
outputBias = b(offset+1:offset+nLabels);

f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    gbInput = zeros(size(inputBias));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
       gbHidden{h-1} = zeros(size(hiddenBias{h-1}));
    end
    gOutput = zeros(size(outputWeights));
    gbOutput = zeros(size(outputBias));
end

ip{1} = X * inputWeights+ inputBias'*nInstances;
fp{1} = tanh(ip{1});
for h = 2:length(nHidden)
    ip{h} = fp{h-1}*hiddenWeights{h-1}+ hiddenBias{h-1}';
    fp{h} = tanh(ip{h});
end % Forward
yhat = fp{end}*outputWeights+ outputBias';


if nargout > 1
    pyhat = exp(yhat) / sum(exp(yhat));
    True = ( y == 1);
    err = pyhat - True;
    gOutput = fp{end}'* err;
    gbOutput=err;
    if length(nHidden) > 1
        % Last Layer of Hidden Weights
        clear backprop
        for c = 1:nLabels
            backprop(c,:) = err(c)*(sech(ip{end}).^2.*outputWeights(:,c)');
            gHidden{end} = gHidden{end} + fp{end-1}'*backprop(c,:);
            gbHidden{end} = gbHidden{end} + backprop(c, :)';
        end
        backprop = sum(backprop,1);
        % Other Hidden Layers
        for h = length(nHidden)-2:-1:1
            backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
            gHidden{h} = gHidden{h} + fp{h}'*backprop;
        end
    else
        for c = 1:nLabels
            backprop(c,:) = err(c) * (sech(ip{end}) .^ 2 .* outputWeights(:,c)');
            gbInput = gbInput + backprop(c, :)';
        end
        % Input Weights
        gInput = X'*err*outputWeights'.*(sech(ip{end}).^2);
    end
end
    

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    gb = zeros(size(b));
    g(1:nVars*nHidden(1)) = gInput(:);
    gb(1:nHidden(1)) = gbInput(:);
    offset = nVars*nHidden(1);
    offsetb = nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        gb(offsetb+1:offsetb+nHidden(h)) = gbHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
        offsetb = offsetb + nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
    gb(offsetb+1:offsetb+nLabels) = gbOutput(:);
end