close all;

%load training data
fileName = 'training_pitches2.csv';
dataAndID1 = csvread(fileName);
%separate traning data and data labels
dataTrain = dataAndID1(:,1:end-1);
labelTrain = dataAndID1(:,end);

%load testing data
fileName = 'testing_pitches2.csv';
dataAndID2 = csvread(fileName);
%separate traning data and data labels
dataTest = dataAndID2(:,1:end-1);
labelTest = dataAndID2(:,end);

%get the unique class label list
uniqueID = unique(labelTrain);

%task 3, nonparametric method for Bayes classifier design
Post_Matrix = []; % initialize with zeros

for i = 1:size(uniqueID) % iterate over each class
    y = []; % initialize with zeros
    data = dataTrain([labelTrain == uniqueID(i)],:);
    m = mean(data,1);
    if size(data,1) == 1
        var = zeros(9);
    else
        var = cov(data);
    end
    % add 0.001 to the diagonal if matrix is not invertible
    if det(var) == 0
        v = [0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001];
        A = diag(v);
        var = var + A;
    else
        var = var;
    end
    varNaive = diag(diag(var)); % get the diagnal components of var and set other components to zero

    for i = 1:size(data,2)
        y = [y getGaussbyParametric(dataAndID1(:,i),m(1,i),varNaive(i,i))]; % call the function to calculate the Gaussian of each point
    end
    
    % calculate P(x|C)
    likelihood = prod(y,2);

    % calculate P(C)
    Prob_C = size(data,1)./size(dataAndID1,1);

    % calculate posterior probability
    Post_Prob = (likelihood) .* (Prob_C);

    % put posterior probabilities into a matrix
    Post_Matrix = [Post_Matrix Post_Prob];
    
end

%get the predicted class ID
predictedIDmatrix = []; % initialize

for i = 1:size(dataAndID1,1)
   [value, predictedID] = max(Post_Matrix(i,:));
   predictedIDmatrix = [predictedIDmatrix predictedID]; % each column is the predicted class ID 
end

%compute accuracy
accuracy = sum(predictedIDmatrix' == labelTest)/length(labelTest)
