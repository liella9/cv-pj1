%L2正则化+softmax
load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [200];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
%仿照W生成步骤生成b
nb = nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
    nb = nb+nHidden(1);
end
nParams = nParams+nHidden(end)*nLabels;
nb=nb+nLabels;
w = randn(nParams,1);
b = randn(nb,1);
wbest=w;
bbest=b;
validbest=1;
stepSize = 1e-2;
% Train with stochastic gradient
maxIter = 100000;
beta = 0.9;
lambda=5e-3;
loss=[];
loss2=[];
% softmax代替平方损失
funObj = @(w,b,i)MLPclassificationLoss6(w,b,X(i,:),yExpanded(i,:),nHidden,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict6(w,b,Xvalid,nHidden,nLabels);
        yhat2 = MLPclassificationPredict6(w,b,X,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        lossi=sum(yhat~=yvalid)/t;
        loss2i=sum(yhat2~=y)/t;
        loss=[loss,lossi];
        loss2=[loss2,loss2i];
        if sum(yhat~=yvalid)/t<0.06
            stepSize=stepSize*0.5;
        end
    end
    if sum(yhat~=yvalid)/t<validbest
        validbest=sum(yhat~=yvalid)/t;
        wbest=w;
        bbest=b;
    end
    i = ceil(rand*n);
    [f,g,gb] = funObj(w,b,i);
    w_temp = w; % Store the weight of the last iteration
    w = w - stepSize * (g+lambda*w) ; 
    b = b - stepSize * gb ;
end
% Evaluate test error
yhat = MLPclassificationPredict6(wbest,bbest,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
plot(1:20,log(loss));hold on; plot(1:20,log(loss2));
legend('validation','training','testing');