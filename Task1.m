mat1 = struct2cell(load('parkinson-Training'));
training = mat1{1};

% Naive Bayes algorithm
% STEP 1: Estimating pi(k) for k = 1 and k = 0

count_y = 0;
count_y0 = 0;

for i=1:size(training, 1)
    if(training(i,23) == 1)
        count_y = count_y + 1;
    end
    if(training(i,23) == 0)
        count_y0 = count_y0 + 1;
    end        
end
pi_0 = count_y0/(count_y0+count_y);
pi_1 = count_y/(count_y0+count_y);


% Computing these for all the inputs
% count(xij ^ y = k)/count(y = k)
%evaluate mean and std for all the variables for the two classes 
%{yes,no} == {1,0}. yes -> has parkinson no -> does not have parkinson

% For simplicity the training array has been filtered for yes's and no's
% and now the mean and std can easily be evaluated for the two classes
training1 = ones(114, 23);
training0 = ones(42, 23);

count1 = 1; 
count0 = 1;
for k=1:156
    if training(k, 23) == 1
        training1(count1,:) = training(k,:);
        count1 = count1 + 1;
    else
        training0(count0,:) = training(k,:);
        count0 = count0 + 1;
    end
end

count1 = count1 - 1;
count0 = count0 - 1;
% Decrement the variables to keep track.
meanYes = zeros(1,23);
stdYes = zeros(1,23);
meanNo = zeros(1,23);
stdNo = zeros(1,23);

for i=1:22
    meanYes(1,i) = mean(training1(:,i));
    meanNo(1,i) = mean(training0(:,i));
    stdYes(1,i) = std(training1(:,i));
    stdNo(1,i) = std(training0(:,i));
end

% We can say at this point that the training set has been learnt
% in the form of the parameters of the gaussian distribution
% lets now classify the dataset using the gaussian function implemented
% by me and label the test data set 
    
mat2 = struct2cell(load('parkinson-Test'));
test = mat2{1};

%time to consider the independence of the probabilities of the features
results = zeros(39, 1);
val = 1;
yes = 1;
no = 1;
probMat = ones(2, 39);
for i=1:39
    yes = 1;
    no = 1;
    for j=1:22
        tx = normpdf(test(i,j),meanYes(1,j), stdYes(1,j));
        yes = yes*tx;
        ty = normpdf(test(i,j),meanNo(1,j), stdNo(1,j));
        no = no*ty;
    end
%     Now we have to mutliply yes and no with pi_1 and pi_0 which were
%     evaluated earlier

yes = yes*pi_1;
no = no*pi_0;

%   Normalize yes and no
    nYes = yes/(yes+no);
    nNo = no/(yes+no);
    probMat(1,i) = nYes;
    probMat(2,i) = nNo;
    alpha = nYes/nNo;
    
%     This is the classification criteria
    if alpha >= 1
        results(i,1) = 1;
    else
        results(i,1) = 0;
    end
    
end
writetable(cell2table(num2cell(results)), 'output.txt');
plotconfusion(transpose(test(:, 23)),probMat(1,:));
pause;
plotroc(transpose(test(:, 23)),probMat(1,:));
