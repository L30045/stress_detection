%% create a spatial filter using all channels
% load
inc_list_lib = cell(5,1);
norm_list_lib = cell(5,1);
% cd('\\hoarding\yuan\Documents\2019 stress in classroom\resting collection\bp_refA_rmBadCh\ASR\ICrej')
load('icawm_cls0.mat','icawm_cls0_inc','icawm_cls0_norm','chan30','chanlocs30','inc_list','norm_list');
inc_list_lib{1} = inc_list;
norm_list_lib{1} = norm_list;
load('icawm_cls1.mat','icawm_cls1_inc','icawm_cls1_norm','inc_list','norm_list');
inc_list_lib{2} = inc_list;
norm_list_lib{2} = norm_list;
load('icawm_cls2.mat','icawm_cls2_inc','icawm_cls2_norm','inc_list','norm_list');
inc_list_lib{3} = inc_list;
norm_list_lib{3} = norm_list;
load('icawm_cls3.mat','icawm_cls3_inc','icawm_cls3_norm','inc_list','norm_list');
inc_list_lib{4} = inc_list;
norm_list_lib{4} = norm_list;
load('icawm_cls4.mat','icawm_cls4_inc','icawm_cls4_norm','inc_list','norm_list');
inc_list_lib{5} = inc_list;
norm_list_lib{5} = norm_list;
% load('\\hoarding\yuan\Documents\2019 stress in classroom\resting collection\bp_refA_rmBadCh\ASR\label.mat')
icawm_lib_inc = {icawm_cls0_inc, icawm_cls1_inc, icawm_cls2_inc, icawm_cls3_inc, icawm_cls4_inc};
icawm_lib_norm = {icawm_cls0_norm, icawm_cls1_norm, icawm_cls2_norm, icawm_cls3_norm, icawm_cls4_norm};

%% separate train/ test recording first
%% generate Fold data
nfold = 5;
step_inc = ceil(length(label_increase)/nfold);
step_norm = ceil(length(label_normal)/nfold);
% shuffle
shuffled_label_inc = label_increase(randperm(length(label_increase)));
shuffled_label_norm = label_normal(randperm(length(label_normal)));



for fold_idx = 1:nfold
    if fold_idx*step_inc <= length(label_increase)
        test_inc_idx = shuffled_label_inc((fold_idx-1)*step_inc+1:fold_idx*step_inc);
        test_norm_idx = shuffled_label_norm((fold_idx-1)*step_norm+1:fold_idx*step_norm);
    else
        test_inc_idx = shuffled_label_inc(end-step_inc+1:end);
        test_norm_idx = shuffled_label_norm(end-step_norm+1:end);
    end
    gen_fold_data(icawm_lib_inc, icawm_lib_norm, inc_list_lib, norm_list_lib, test_inc_idx, test_norm_idx, label_increase, label_normal, chan30, fold_idx)
end

disp('Done')

%% generate Leave-one-session-out data
for sess_i = [label_increase', label_normal']
    gen_looSess_data(icawm_lib_inc, icawm_lib_norm, inc_list_lib, norm_list_lib, label_increase, label_normal, sess_i, chan30, {'FZ','FCZ','CZ'});
end
disp('Done')

%% LDA
load('icawm_allCh_fold1.mat');

%% train/ test split
f_feature = 5:9;
f_feature = f_feature+1; % spectra starts from 0
% subj_list = summary_table.SubjectNumber;
inc_data = plt_medall_inc(:,f_feature,:);
inc_data = reshape(inc_data,size(inc_data,1),[]);
norm_data = plt_medall_norm(:,f_feature,:);
norm_data = reshape(norm_data,size(norm_data,1),[]);

% lda
testinc = ismember(label_increase,test_inc_idx);
testnorm = ismember(label_normal,test_norm_idx);
X_train = [inc_data(~testinc,:);norm_data(~testnorm,:)];
Y_train = [ones(sum(~testinc),1); zeros(sum(~testnorm),1)];
X_test= [inc_data(testinc,:);norm_data(testnorm,:)];
Y_test= [ones(sum(testinc),1); zeros(sum(testnorm),1)];
% weight LDA according to number of samples
weights = [ones(sum(~testinc),1)*size(X_train,1)/sum(~testinc);...
    ones(sum(~testnorm),1)*size(X_train,1)/sum(~testnorm);];
% weights = ones(size(X_train,1),1);
clf_lda = fitcdiscr(X_train,Y_train,'weights',weights);
predictResult = predict(clf_lda,X_test);
predTrain = predict(clf_lda,X_train);
figure
cmTrain = confusionchart(Y_train, predTrain);
cmTrain.ColumnSummary = 'column-normalized';
cmTrain.RowSummary = 'row-normalized';
figure
cm = confusionchart(Y_test, predictResult);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
inc_acc = sum(predictResult&Y_test)/sum(predictResult);
norm_acc = sum(~predictResult&~Y_test)/sum(~predictResult);
balanced_acc = (inc_acc+norm_acc)/2;

disp('-------------------')
fprintf('Test Balanced ACC = %2.1f%%\n',balanced_acc*100);
fprintf('Test INC ACC = %2.1f%%\n',inc_acc*100);
fprintf('Test NORM ACC = %2.1f%%\n',norm_acc*100);
fprintf('Train Balanced ACC = %2.1f%%\n',(sum(predTrain&Y_train)/sum(predTrain)+...
                 sum(~predTrain&~Y_train)/sum(~predTrain))/2*100);
fprintf('Train INC ACC = %2.1f%%\n',sum(predTrain&Y_train)/sum(predTrain)*100);
fprintf('Train NORM ACC = %2.1f%%\n',sum(~predTrain&~Y_train)/sum(~predTrain)*100);
disp('-------------------')
% disp('-------------------')
% fprintf('Best balanced ACC = %2.1f%%\n',max(balanced_acc)*100);
% fprintf('Worst balanced ACC = %2.1f%%\n',min(balanced_acc)*100);
% fprintf('Avg. balanced ACC = %2.1f%%\n',mean(balanced_acc)*100);
% % bar(inc_acc)
% [~,maxi] = max(inc_acc);
% [~,mini] = min(inc_acc);
test_inc_max = subj_list(test_inc_idx);
train_inc_max = subj_list(label_increase(~testinc));
test_norm_max = subj_list(test_norm_idx);
train_norm_max = subj_list(label_normal(~testnorm));
% fprintf('MAX ACC # of no repeat = %d\n',sum(~ismember(test_inc_max,train_inc_max)));
% test_inc_min = subj_list(label_increase(subj_choose{1,mini}));
% train_inc_min = subj_list(label_increase(~subj_choose{1,mini}));
% fprintf('MIN ACC # of no repeat = %d\n',sum(~ismember(test_inc_min,train_inc_min)));

figure
histogram(test_inc_max,'DisplayName','Test');
hold on
grid on
histogram(train_inc_max,'DisplayName','Train');
legend
title(sprintf('MAX INC ACC %2.1f%%',max(inc_acc)*100))
figure
histogram(test_norm_max,'DisplayName','Test');
hold on
grid on
histogram(train_norm_max,'DisplayName','Train');
legend
title(sprintf('MAX NORM ACC %2.1f%%',max(norm_acc)*100))

%% visualize spatial filters
for f_i = 1:nfold
    savefigpath = sprintf('cls_idx_k5/fold_data/fold%d/',f_i);
    if ~exist(savefigpath,'dir')
        mkdir(savefigpath)
    end
    load(sprintf('icawm_allCh_fold%d.mat',f_i));
    figure
    for cls_i = 1:5
        topoplot(icawm_lib_med(:,cls_i),chanlocs30);
        title(sprintf('Cls %d',cls_i-1))
        saveas(gcf, [savefigpath,sprintf('cls%d_spf_med.png',cls_i-1)]);
        clf
    end
    close(gcf)
end

%% record acc for Fold data
filename_lib = 1:5;
f_feature = 4:8;
[balanced_acc, inc_acc, norm_acc, pos_pred_val, neg_pred_val] = vis_LDA_acc(filename_lib, f_feature, 'fold');
% -------------------
% Avg. Test Balanced ACC = 82.0%
% -------------------
% Avg. Test INC ACC = 80.0%
% -------------------
% Avg. Test NORM ACC = 84.0%
% -------------------
% Avg. Pos. Pred. Val.= 63.9%
% -------------------
% Avg. Neg. Pred. Val.= 93.0%
% -------------------
% Avg. Train INC ACC = 96.2%
% -------------------
% Avg. Train NORM ACC = 92.5%
% -------------------


%% record acc for LOO Session
filename_lib = [label_increase',label_normal'];
f_feature = 4:8;
[balanced_acc, inc_acc, norm_acc, pos_pred_val, neg_pred_val, mis_classified_file] = vis_LDA_acc(filename_lib, f_feature, 'looSess');
% -------------------
% Avg. Test Balanced ACC = 80.3%
% -------------------
% Avg. Test INC ACC = 76.2%
% -------------------
% Avg. Test NORM ACC = 84.5%
% -------------------
% Avg. Pos. Pred. Val.= 59.3%
% -------------------
% Avg. Neg. Pred. Val.= 92.3%
% -------------------
% Avg. Train INC ACC = 95.2%
% -------------------
% Avg. Train NORM ACC = 91.8%
% -------------------
% mis_classified_file = [1,17,27,52,63,12,15,16,24,32,33,39,54,72,83,98];
% mis_cls_file_subj = [1,3,4,9,11,2,3,3,4,5,5,6,9,12,14,16];
mis_cls_file_subj = subj_list(mis_classified_file);
subj_mis_ratio = zeros(4, length(unique(mis_cls_file_subj)));
subj_mis_ratio(1,:) = unique(mis_cls_file_subj);
for subj_i = 1:size(subj_mis_ratio,2)
    subj_mis_ratio(2,subj_i) = sum(subj_list==subj_mis_ratio(1,subj_i));
    subj_mis_ratio(3,subj_i) = sum(mis_cls_file_subj==subj_mis_ratio(1,subj_i));
end
subj_mis_ratio(4,:) = subj_mis_ratio(3,:)./subj_mis_ratio(2,:);



%% visualize Cls PSD
for f_i = 1:nfold
    savefigpath = sprintf('cls_idx_k5/fold_data/fold%d/',f_i);notice that subje
    load(sprintf('icawm_allCh_fold%d.mat',f_i));
    testinc = ismember(label_increase,test_inc_idx);
    testnorm = ismember(label_normal,test_norm_idx);
    for plt_cls = 1:5
        plt_inc = plt_medall_inc(~testinc,:,plt_cls);
        plt_norm = plt_medall_norm(~testnorm,:,plt_cls);
        ranksum_p = zeros(1,50); 
        ttest_h = false(1,50);
        myboot_p = zeros(1,50);
        myboot_h = false(1,50);
        ci_inc = zeros(50,2);
        ci_norm = zeros(50,2);

        for freq = 1:50
            ranksum_p(freq) = ranksum(plt_inc(:,freq),plt_norm(:,freq));
            ttest_h(freq) = ttest2(plt_inc(:,freq),plt_norm(:,freq));
            [myboot_p(freq), myboot_h(freq),~,~,ci_inc(freq,:),ci_norm(freq,:)] = myboot(plt_inc(:,freq),plt_norm(:,freq));
        end

        plt_f = 0:49;
        figure
        solid_line = @(x) mean(x, 'omitnan');
        set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
        h = myShadedErrorBar(plt_f,plt_inc,{solid_line,ci_inc},{'r-','DisplayName','Increase','linewidth',3});
        h.patch.FaceAlpha = 0.3;
        hold on
        grid on
        h = myShadedErrorBar(plt_f,plt_norm,{solid_line,ci_norm},{'b-','DisplayName','Normal','linewidth',3});
        h.patch.FaceAlpha = 0.3;
        % plot significance
        plot(plt_f(ranksum_p<=0.05), mean(plt_inc(:,ranksum_p<=0.05)),'ko','Displayname','ranksum','markersize',15,'linewidth',3);
        plot(plt_f(ttest_h), mean(plt_inc(:,ttest_h)),'gx','Displayname','ttest','markersize',15,'linewidth',3);
        plot(plt_f(myboot_h), mean(plt_inc(:,myboot_h)),'md','Displayname','bootci','markersize',15,'linewidth',3);
        legend(findobj(gca,'-regexp','DisplayName', '[^'']'))
        xlabel('Frequency (Hz)')
        ylabel('Power (dB)')
        title(sprintf('Cls %d',plt_cls-1),'interpreter','none');
        set(gca,'fontsize',20)
        saveas(gcf, [savefigpath,sprintf('psd_cls%d_fold%d.png',plt_cls-1,f_i)]);
        close(gcf)
        
    end
end


%% TO DO LIST
% Use frontal channels only to project IC activities
% Shrinkage LDA

%% Use Frontal channels PSD to run LDA
f_feature = 3:7;
% tarCh = {'F3','F4','FZ','F7','F8','FC3','FCZ','FC4'};
% tarCh = {'FZ','FCZ','CZ'};
% tarCh = {'FZ'};
% tarCh = {'CZ','CP3','CP4','FCZ','FZ','PZ'};
tarCh = {'FZ','CZ','proj'};
[balanced_acc, inc_acc, norm_acc, pos_pred_val, neg_pred_val] = vis_LDA_acc(tarCh, f_feature, 'looSess');

% tarCh = {'F3','F4','FZ','F7','F8','FC3','FCZ','FC4'};
% -------------------
% Avg. Test Balanced ACC = 62.0%
% -------------------
% Avg. Test INC ACC = 56.0%
% -------------------
% Avg. Test NORM ACC = 68.0%
% -------------------
% Avg. Pos. Pred. Val.= 35.7%
% -------------------
% Avg. Neg. Pred. Val.= 82.7%
% -------------------
% Avg. Train INC ACC = 100.0%
% -------------------
% Avg. Train NORM ACC = 93.6%
% -------------------

% tarCh = {'Fz'}
% -------------------
% Avg. Test Balanced ACC = 75.3%
% -------------------
% Avg. Test INC ACC = 68.0%
% -------------------
% Avg. Test NORM ACC = 82.7%
% -------------------
% Avg. Pos. Pred. Val.= 66.0%
% -------------------
% Avg. Neg. Pred. Val.= 89.1%
% -------------------
% Avg. Train INC ACC = 77.5%
% -------------------
% Avg. Train NORM ACC = 82.1%
% -------------------

% tarCh = {'CZ','CP3','CP4','FCZ','FZ','PZ'};
% -------------------
% Avg. Test Balanced ACC = 78.7%
% -------------------
% Avg. Test INC ACC = 76.0%
% -------------------
% Avg. Test NORM ACC = 81.3%
% -------------------
% Avg. Pos. Pred. Val.= 59.5%
% -------------------
% Avg. Neg. Pred. Val.= 91.6%
% -------------------
% Avg. Train INC ACC = 100.0%
% -------------------
% Avg. Train NORM ACC = 93.2%
% -------------------


%% Use Frontal channels to project IC activities
% Compare the acc
% FZ, FCZ, CZ in channel domain
f_feature = 4:8;
sess_list = [label_increase',label_normal'];
tarCh = {'FZ','FCZ','CZ'};
bacc_ch = vis_LDA_acc(tarCh, f_feature, 'looSess');
bacc_ic = vis_LDA_acc(sess_list, f_feature, 'looSess');
bacc_ic_proj = vis_LDA_acc(sess_list, f_feature, 'looSess_proj');
% Conclusion: Using 3 channels can already reach the same INC ACC as using IC
% spatial filter. Projected IC spatial filter with only 3 channels will
% drop the performance.

figure
bar([bacc_ch, bacc_ic, bacc_ic_proj]);
set(gca,'xticklabel',{'Ch','IC','IC_{proj}'})
ylabel('Balanced ACC.')
grid on
set(gca,'fontsize',20)


%% visualize FZ FCZ CZ features
tarCh = {'FZ','FCZ','CZ'};
f_feature = 4:8;
tarPSD = cellfun(@(x) x(ismember(chan30, tarCh), f_feature+1), ch_psd_lib, 'uniformoutput',0);
incPSD = tarPSD(1:21);
normPSD = tarPSD(22:end);
rand_test = randi(92,1);
if rand_test <= 21
    inc_train = incPSD;
    inc_train(rand_test) = [];
    norm_train = normPSD;
    X_test = reshape(incPSD{rand_test},1,[]);
    Y_test = true;
else
    norm_train = normPSD;
    norm_train(rand_test-21) = [];
    inc_train = incPSD;
    X_test = reshape(normPSD{rand_test-21},1,[]);
    Y_test = false;
end


X_train = cell2mat([cellfun(@(x) reshape(x,1,[]),inc_train','uniformoutput',0);...
                    cellfun(@(x) reshape(x,1,[]),norm_train','uniformoutput',0)]);
Y_train = [true(size(inc_train{1},2),1);false(size(norm_train{1},2),1)];

% weight LDA according to number of samples
weights = [ones(16,1)*size(X_train,1)/16;...
    ones(56,1)*size(X_train,1)/56;];
% weights = ones(size(X_train,1),1);
clf_lda = fitcdiscr(X_train,Y_train,'weights',weights);
predictResult = predict(clf_lda,X_test);
predTrain = predict(clf_lda,X_train);
figure
cmTrain = confusionchart(Y_train, predTrain);
cmTrain.ColumnSummary = 'column-normalized';
cmTrain.RowSummary = 'row-normalized';
figure
cm = confusionchart(Y_test, predictResult);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
inc_acc = sum(predictResult&Y_test)/sum(predictResult);
norm_acc = sum(~predictResult&~Y_test)/sum(~predictResult);
balanced_acc = (inc_acc+norm_acc)/2;

% visualize LDA results
val_inc = cellfun(@(x) reshape(x,1,[])*clf_lda.Coeffs(2).Linear, incPSD);
val_norm = cellfun(@(x) reshape(x,1,[])*clf_lda.Coeffs(2).Linear, normPSD);
figure
plot(val_inc, 0, 'rx', 'DisplayName','Increase','linewidth',3,'markersize',15)
hold on
grid on
plot(val_norm, 0, 'bo', 'DisplayName','Normal','linewidth',3,'markersize',15)
legend(['Increase',repmat({''},1,20),'Normal'])

% tsne
plt_y = tsne(cell2mat([cellfun(@(x) reshape(x,1,[]),incPSD','uniformoutput',0);...
              cellfun(@(x) reshape(x,1,[]),normPSD','uniformoutput',0)]));
cmap = jet(21);
figure; gscatter(plt_y(1:21,1),plt_y(1:21,2),label_increase);
figure; gscatter(plt_y(22:end,1),plt_y(22:end,2),label_normal);
% gscatter(plt_y(:,1),plt_y(:,2),[ones(21,1);zeros(71,1)]);

%% removed outlier
plt_y_rm = tsne(cell2mat([cellfun(@(x) reshape(x,1,[]),incPSD','uniformoutput',0);...
              cellfun(@(x) reshape(x,1,[]),normPSD([1:12,14:end])','uniformoutput',0)]));
figure;gscatter(plt_y_rm(:,1),plt_y_rm(:,2),[ones(21,1);zeros(70,1)]);
X_train = cell2mat([cellfun(@(x) reshape(x,1,[]),incPSD(1:16)','uniformoutput',0);...
                    cellfun(@(x) reshape(x,1,[]),normPSD([1:12,14:56])','uniformoutput',0)]);
Y_train = [true(16,1);false(55,1)];
X_test = cell2mat([cellfun(@(x) reshape(x,1,[]),incPSD(17:end)','uniformoutput',0);...
                    cellfun(@(x) reshape(x,1,[]),normPSD(57:end)','uniformoutput',0)]);
Y_test = [true(5,1);false(15,1)];
% weight LDA according to number of samples
weights = [ones(16,1)*size(X_train,1)/16;...
    ones(55,1)*size(X_train,1)/55;];
% weights = ones(size(X_train,1),1);
clf_lda = fitcdiscr(X_train,Y_train,'weights',weights);
predictResult = predict(clf_lda,X_test);
predTrain = predict(clf_lda,X_train);
figure
cmTrain = confusionchart(Y_train, predTrain);
cmTrain.ColumnSummary = 'column-normalized';
cmTrain.RowSummary = 'row-normalized';
figure
cm = confusionchart(Y_test, predictResult);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
inc_acc = sum(predictResult&Y_test)/sum(predictResult);
norm_acc = sum(~predictResult&~Y_test)/sum(~predictResult);
balanced_acc = (inc_acc+norm_acc)/2;

%% CSP
tarCh = {'FZ','FCZ','CZ'};
f_feature = 4:8;
tarPSD = cellfun(@(x) x(ismember(chan30, tarCh), f_feature+1), ch_psd_lib, 'uniformoutput',0);
incPSD = tarPSD(1:21);
normPSD = tarPSD(22:end);
inc_data = cell2mat(cellfun(@(x) reshape(x,1,[]),incPSD','uniformoutput',0))';
norm_data = cell2mat(cellfun(@(x) reshape(x,1,[]),normPSD([1:12,14:end])','uniformoutput',0))';
cov_inc = inc_data*inc_data';
cov_norm = norm_data*norm_data';
[V,D] = eig(pinv(cov_inc)*cov_norm);
csp_vec1 = V(:,1);
csp_vec2 = V(:,2);
csp1_inc = csp_vec1'*inc_data;
csp1_norm = csp_vec1'*norm_data;
csp2_inc = csp_vec2'*inc_data;
csp2_norm = csp_vec2'*norm_data;
figure
h1 = scatter(csp1_inc,csp2_inc,100,'rx','DisplayName','Increase');
h1.LineWidth = 3;
hold on
grid on
h2 = scatter(csp1_norm,csp2_norm,100,'bo','DisplayName','Normal');
h2.LineWidth = 3;
legend
xlabel('CSP 1')
ylabel('CSP 2')
set(gca,'fontsize',20)
figure
l_feat = length(f_feature);
bar([abs(csp_vec1(1:l_feat)),abs(csp_vec2(1:l_feat))])
set(gca,'xticklabel',f_feature)
title('FZ')
figure
bar([abs(csp_vec1(l_feat+1:2*l_feat)),abs(csp_vec2(l_feat+1:2*l_feat))])
set(gca,'xticklabel',f_feature)
title('FCZ')
figure
bar([abs(csp_vec1(2*l_feat+1:end)),abs(csp_vec2(2*l_feat+1:end))])
set(gca,'xticklabel',f_feature)
title('CZ')

%% ASR only data results
f_feature = 3:7;
tarCh = {'FZ','FCZ','CZ','raw'};
[balanced_acc, inc_acc, norm_acc, pos_pred_val, neg_pred_val] = vis_LDA_acc(tarCh, f_feature, 'looSess');

% 5 folds
% -------------------
% Avg. Test Balanced ACC = 67.3%
% -------------------
% Avg. Test INC ACC = 56.0%
% -------------------
% Avg. Test NORM ACC = 78.7%
% -------------------
% Avg. Pos. Pred. Val.= 46.6%
% -------------------
% Avg. Neg. Pred. Val.= 84.4%
% -------------------
% Avg. Train INC ACC = 92.5%
% -------------------
% Avg. Train NORM ACC = 86.8%
% -------------------

% LOO
% -------------------
% Avg. Test Balanced ACC = 70.7%
% -------------------
% Avg. Test INC ACC = 66.7%
% -------------------
% Avg. Test NORM ACC = 74.6%
% -------------------
% Avg. Pos. Pred. Val.= 43.8%
% -------------------
% Avg. Neg. Pred. Val.= 88.3%
% -------------------
% Avg. Train INC ACC = 90.4%
% -------------------
% Avg. Train NORM ACC = 79.3%
% -------------------

%% ASR+ICA
bal_acc_lib = zeros(5,100);
inc_acc_lib = zeros(5,100);
norm_acc_lib = zeros(5,100);
pos_pred_val_lib = zeros(5,100);
neg_pred_val_lib = zeros(5,100);

for b_i = 1:100
    [b, inc_acc, norm_acc, pos_pred_val, neg_pred_val] = vis_LDA_acc(tarCh, f_feature, 'fold');
    bal_acc_lib(:,b_i) = b;
    inc_acc_lib(:,b_i) = inc_acc;
    norm_acc_lib(:,b_i) = norm_acc;
    pos_pred_val_lib(:,b_i) = pos_pred_val;
    neg_pred_val_lib(:,b_i) = neg_pred_val;
end

%%
plt_lib = pos_pred_val_lib;
tname = 'Pos.Pred.Val.';

figure
histogram(mean(plt_lib,1),'binwidth',0.01,'DisplayName', 'Count');
hold on
yline(0,'DisplayName',sprintf('mean = %f',mean(plt_lib,'all')));
yline(0,'DisplayName',sprintf('median = %f',median(mean(plt_lib,1))));
legend
xlabel('Probability')
ylabel('Count')
title(sprintf('Mean %s in 100 times 5 fold validation',tname))

figure
histogram(plt_lib(:),'binwidth',0.01,'DisplayName', 'Count');
hold on
yline(0,'DisplayName',sprintf('mean = %f',mean(plt_lib,'all')));
yline(0,'DisplayName',sprintf('median = %f',median(plt_lib,'all')));
legend
xlabel('Probability')
ylabel('Count')
title(sprintf('%s in 100 times 5 fold validation',tname))

%% ACC Comparison
% LOO RAW
% -------------------
% Avg. Test Balanced ACC = 62.8%
% -------------------
% Avg. Test INC ACC = 52.4%
% -------------------
% Avg. Test NORM ACC = 73.2%
% -------------------
% Avg. Pos. Pred. Val.= 36.7%
% -------------------
% Avg. Neg. Pred. Val.= 83.9%
% -------------------
% Avg. Train INC ACC = 82.9%
% -------------------
% Avg. Train NORM ACC = 79.9%
% -------------------

% LOO ASR
% -------------------
% Avg. Test Balanced ACC = 70.7%
% -------------------
% Avg. Test INC ACC = 66.7%
% -------------------
% Avg. Test NORM ACC = 74.6%
% -------------------
% Avg. Pos. Pred. Val.= 43.8%
% -------------------
% Avg. Neg. Pred. Val.= 88.3%
% -------------------
% Avg. Train INC ACC = 90.4%
% -------------------
% Avg. Train NORM ACC = 79.3%
% -------------------


% LOO ASR+ICA
% -------------------
% Avg. Test Balanced ACC = 78.2%
% -------------------
% Avg. Test INC ACC = 76.2%
% -------------------
% Avg. Test NORM ACC = 80.3%
% -------------------
% Avg. Pos. Pred. Val.= 53.3%
% -------------------
% Avg. Neg. Pred. Val.= 91.9%
% -------------------
% Avg. Train INC ACC = 90.5%
% -------------------
% Avg. Train NORM ACC = 86.7%
% -------------------


% LOO ASR+Proj_rmEye
% -------------------
% Avg. Test Balanced ACC = 77.3%
% -------------------
% Avg. Test INC ACC = 71.4%
% -------------------
% Avg. Test NORM ACC = 83.1%
% -------------------
% Avg. Pos. Pred. Val.= 55.6%
% -------------------
% Avg. Neg. Pred. Val.= 90.8%
% -------------------
% Avg. Train INC ACC = 81.2%
% -------------------
% Avg. Train NORM ACC = 86.1%
% -------------------
