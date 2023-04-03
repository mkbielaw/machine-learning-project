clear; close all; clc;
load('E:\Studia\L''aquila\WC\Project\data\2K_nn_200Hz\40K.mat');

%1 - rayleigh, 2 - rician

%SAME SEED, 80% / 20% RATIO
rayClass = [rayleigh09,zeros(size(rayleigh09,1),1)];
rayClass(:,2) = rayClass(:,2)+1;
ricClass = [rician00,ones(size(rician00,1),1)];
ricClass(:,2) = ricClass(:,2)+1;

rayTrain = rayClass(1:floor(0.8*length(rayClass)),:);
ricTrain = ricClass(1:floor(0.8*length(ricClass)),:);

rayTest = rayClass(floor(0.8*length(rayClass)):end,:);
ricTest = ricClass(floor(0.8*length(ricClass)):end,:);

X_train = [rayTrain(:,1); ricTrain(:,1)];
Y_train = [rayTrain(:,2); ricTrain(:,2)];

X_test = [rayTest(:,1); ricTest(:,1)];
Y_test = [rayTest(:,2); ricTest(:,2)];

%PROBABILISTIC GENERATIVE START____________________________________________
%{
% Estimate class priors
class_priors = tabulate(Y_train);
class_priors = class_priors(:,3)/100;

% Estimate class-conditional densities
class_conditional_densities = cell(2,1);
%Rayleigh
class_conditional_densities{1} = fitdist(X_train(Y_train == 1,:),'Rayleigh');
%Rician
class_conditional_densities{2} = fitdist(X_train(Y_train == 2,:),'Rician');

% Apply Bayes' rule for classification
Y_pred = zeros(size(X_test,1),1);
for i = 1:size(X_test,1)
    posterior_probs = zeros(2,1);
    for j = 1:2
        posterior_probs(j) = class_priors(j)*pdf(class_conditional_densities{j},X_test(i,:));
    end
    [~,Y_pred(i)] = max(posterior_probs);
end

evaluate(Y_test,Y_pred);
%}
%PROBABILISTIC GENERATIVE END______________________________________________




% FORMING LAG MATRIX
%{
Xt = [X_train; X_test];
Yt = [Y_train; Y_test];
X = [];
Y = [];
dx = 2;

    temp_x = lagmatrix(Xt,dx:-1:0);
    temp_x(1:dx,:) = [];
    temp_y = Yt(dx+1:end);
    X = [X; temp_x];
    Y = [Y; temp_y];

X_train = X(1:floor(0.8*length(X)),:);
Y_train = Y(1:floor(0.8*length(Y)),:);

X_test = X(floor(0.8*length(X)):end,:);
Y_test = Y(floor(0.8*length(Y)):end,:);
%}




%CLASSIFICATION TREE START_________________________________________________
%{
tree = fitctree(X_train,Y_train);
%view(tree,'Mode','graph');
%259 optimal MinLeafSize
%tree = fitctree(X_train,Y_train,MaxNumSplits=10,MinLeafSize=259);
Y_pred = predict(tree,X_test);
for i=1:size(Y_test,1)
    testCT(i) = Y_test(i,1) == Y_pred(i,1);
end
evaluate(Y_test,Y_pred);
%disp(['CT correct guess = ', num2str(sum(testCT)/size(Y_test,1)*100), '%'])
%}
%CLASSIFICATION TREE END___________________________________________________




%RANDOM FOREST START_______________________________________________________
%{
for numTrees = [1 2 5 10 20 50]
    forest = TreeBagger(numTrees,X_train,Y_train);
    Y_pred = predict(forest,X_test);
    Y_pred = str2double(Y_pred);
    disp(['Evaluation for ', num2str(numTrees),' trees:'])
    evaluate(Y_test,Y_pred);
    disp('-')
    for i=1:size(Y_test,1)
        testRF(i) = Y_test(i,1) == Y_pred(i,1);
    end
    %disp(['RF correct guess with ', num2str(numTrees), ' = ', num2str(sum(testRF)/size(Y_test,1)*100), '%'])
end
%}
%RANDOM FORESTS END _______________________________________________________




%MULTINOMIAL LOGISTIC REGRESSION START_____________________________________
%{
MnrMdl = mnrfit(X_train,Y_train);
classHatMnr = mnrval(MnrMdl,X_test);
Y_pred = zeros(size(X_test,1),1);
for i = 1:size(X_test,1)
    [~,Y_pred(i)] = max(classHatMnr(i,:));
end
evaluate(Y_test,Y_pred);
%}
%MULTINOMIAL LOGISTIC REGRESSION END_______________________________________




%SUPPORT VECTOR MACHINES START_____________________________________________

%compare different kernels
%{
for i=1:3
switch i
    case 1
        kernel = {'gaussian','',''};
    case 2
        kernel = {'linear','',''};
    case 3
        kernel = {'polynomial','PolynomialOrder',2};
end

SvmMdl = fitcsvm(X_train,Y_train,'KernelFunction',kernel{1});
Y_pred = predict(SvmMdl,X_test);

% Display results
disp([kernel{1},':'])
evaluate(Y_test,Y_pred);
disp('-')
end
%}
%SUPPORT VECTOR MACHINES END_______________________________________________




%NEURAL NETWORKS START_____________________________________________________
%{
Mdl=fitcnet(X_train,Y_train,Activations="sigmoid");
%Mdl=fitcnet(X_train,Y_train,'OptimizeHyperparameters','auto');
Y_pred=predict(Mdl,X_test);
evaluate(Y_test,Y_pred);
%}
%NEURAL NETWORKS END_______________________________________________________




%{
%LAGGED BAYES
X_lagged = [];
Y_lagged = [];
dx = 2;

    temp_x = lagmatrix(X_train,dx:-1:0);
    temp_x(1:dx,:) = [];
    temp_y = Y_train(dx+1:end);
    X_lagged = [X_lagged; temp_x];
    Y_lagged = [Y_lagged; temp_y];

% Estimate class priors
class_priors = tabulate(Y_train);
class_priors = class_priors(:,3)/100;

% Estimate class-conditional densities
class_conditional_densities = cell(2,1);
%Rayleigh
class_conditional_densities{1} = fitdist(X_lagged(Y_lagged == 1,:),'Rayleigh');
%Rician
class_conditional_densities{2} = fitdist(X_lagged(Y_lagged == 2,:),'Rician');

% Apply Bayes' rule for classification
X_lagged_test = [];
Y_lagged_test = [];
dx = 2;

    temp_x = lagmatrix(X_test,dx:-1:0);
    temp_x(1:dx,:) = [];
    temp_y = Y_test(dx+1:end);
    X_lagged_test = [X_lagged_test; temp_x];
    Y_lagged_test = [Y_lagged_test; temp_y];

Y_pred = zeros(size(X_lagged_test,1),1);
for i = 1:size(X_lagged_test,1)
posterior_probs = zeros(2,1);
for j = 1:2
posterior_probs(j) = class_priors(j)*pdf(class_conditional_densities{j},X_lagged_test(i,:));
end
[~,Y_pred(i)] = max(posterior_probs);
end

evaluate(Y_lagged_test,Y_pred);
%}





function evaluate(Y_test,Y_pred)
% Compute confusion matrix
cm = confusionmat(Y_test,Y_pred);

% Compute accuracy
acc = sum(diag(cm))/sum(cm(:));

% Compute precision, sensitivity (recall), and specificity
prec = cm(2,2)/(cm(2,2) + cm(1,2));
sens = cm(2,2)/(cm(2,2) + cm(2,1));
spec = cm(1,1)/(cm(1,1) + cm(2,1));

% Compute F1 score
f1 = (((prec)^-1+(sens)^-1)/2)^-1;

% Display results
fprintf('Accuracy: %.4f%%\n',acc*100);
fprintf('Precision: %.4f%%\n',prec*100);
fprintf('Sensitivity: %.4f%%\n',sens*100);
fprintf('Specificity: %.4f%%\n',spec*100);
fprintf('F1 Score: %.4f%%\n',f1*100);
end