
% load mnist data
d = load('mnist_classify.mat');

fea = [d.data; d.testdata];
lab = [d.labels; d.testlabels];

% save to file
save('mnist_fea.txt', '-ascii', 'fea');
save('mnist_lab.txt', '-ascii', 'lab');