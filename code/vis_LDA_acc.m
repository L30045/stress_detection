function [balanced_acc, inc_acc, norm_acc, pos_pred_val, neg_pred_val, varargout] = vis_LDA_acc(filename_lib, f_feature, vali_method)
addpath(genpath('/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/ASR/ICrej/cls_idx_k5/'));
%% Channel based or IC based
if isa(filename_lib,'double')
    nfold = length(filename_lib);
else
    if strcmp(filename_lib{end},'ASR')
        load('ch_power_asr.mat',...
            'ch_psd_lib','chan30','label_increase','label_normal');
        filename_lib(end) = [];
    elseif strcmp(filename_lib{end},'proj')
        load('ch_power_proj.mat',...
            'ch_psd_lib','chan30','label_increase','label_normal');
        filename_lib(end) = [];
    elseif strcmp(filename_lib{end},'raw')
        load('ch_power_raw.mat',...
            'ch_psd_lib','chan30','label_increase','label_normal');
        filename_lib(end) = [];
    else
        load('ch_power.mat',...
            'ch_psd_lib','chan30','label_increase','label_normal');
    end
    tarCh_idx = ismember(chan30,filename_lib);
    inc_data = cell2mat(cellfun(@(x) reshape(x(tarCh_idx,f_feature+1),1,[]),ch_psd_lib(1:length(label_increase)),'uniformoutput',0)');
    norm_data = cell2mat(cellfun(@(x) reshape(x(tarCh_idx,f_feature+1),1,[]),ch_psd_lib(length(label_increase)+1:end),'uniformoutput',0)');
    % remove outlier: recording 23, based on TSNE plot
%     norm_data = norm_data([1:12,14:end],:);
%     label_normal(label_normal==23) = [];
    
    if strcmp(vali_method, 'fold')
        nfold = 5;
        [test_inc_lib, test_norm_lib] = split_fold(nfold,label_increase,label_normal);
    else
        nfold = length(ch_psd_lib);
        % remove outlier: recording 23, based on TSNE plot
%         nfold = nfold-1;
    end
end
f_feature = f_feature+1; % spectra starts from 0
balanced_acc = zeros(1,nfold);
inc_acc = zeros(1,nfold);
norm_acc = zeros(1,nfold);
inc_acc_tr = zeros(1,nfold);
norm_acc_tr = zeros(1,nfold);
pos_pred_val = zeros(1,nfold);
neg_pred_val = zeros(1,nfold);
pred = zeros(1,nfold);

for f_i = 1:nfold
    % lda
    switch vali_method
        case 'fold'
            if isa(filename_lib,'double')
                load(sprintf('icawm_allCh_fold%d.mat',filename_lib(f_i)),...
                    'plt_medall_inc','plt_medall_norm','test_inc_idx','test_norm_idx','label_increase','label_normal');
                inc_data = plt_medall_inc(:,f_feature,:);
                inc_data = reshape(inc_data,size(inc_data,1),[]);
                norm_data = plt_medall_norm(:,f_feature,:);
                norm_data = reshape(norm_data,size(norm_data,1),[]);
                testinc = ismember(label_increase,test_inc_idx);
                testnorm = ismember(label_normal,test_norm_idx);
            else
                testinc = ismember(label_increase, test_inc_lib{f_i});
                testnorm = ismember(label_normal, test_norm_lib{f_i});
            end
        case 'looSess'
            if isa(filename_lib,'double')
                load(sprintf('icawm_allCh_looSess%d.mat',filename_lib(f_i)),...
                    'plt_medall_inc','plt_medall_norm','test_sess','label_increase','label_normal');
                inc_data = plt_medall_inc(:,f_feature,:);
                inc_data = reshape(inc_data,size(inc_data,1),[]);
                norm_data = plt_medall_norm(:,f_feature,:);
                norm_data = reshape(norm_data,size(norm_data,1),[]);
                testinc = ismember(label_increase,test_sess);
                testnorm = ismember(label_normal,test_sess);
            else
                testinc = false(1,size(inc_data,1));
                testnorm = false(1,size(norm_data,1));
                if f_i <= size(inc_data,1)
                    testinc(f_i) = true;
                else
                    testnorm(f_i-size(inc_data,1)) = true;
                end
            end
        case 'looSess_proj'
            if isa(filename_lib,'double')
                load(sprintf('icawm_proj_looSess%d.mat',filename_lib(f_i)),...
                    'plt_medall_inc','plt_medall_norm','test_sess','label_increase','label_normal');
                inc_data = plt_medall_inc(:,f_feature,:);
                inc_data = reshape(inc_data,size(inc_data,1),[]);
                norm_data = plt_medall_norm(:,f_feature,:);
                norm_data = reshape(norm_data,size(norm_data,1),[]);
                testinc = ismember(label_increase,test_sess);
                testnorm = ismember(label_normal,test_sess);
            else
                testinc = false(1,size(inc_data,1));
                testnorm = false(1,size(norm_data,1));
                if f_i <= size(inc_data,1)
                    testinc(f_i) = true;
                else
                    testnorm(f_i-size(inc_data,1)) = true;
                end
            end
    end
    
    % train/ test split
    X_train = [inc_data(~testinc,:);norm_data(~testnorm,:)];
    Y_train = [ones(sum(~testinc),1); zeros(sum(~testnorm),1)];
    X_test= [inc_data(testinc,:);norm_data(testnorm,:)];
    Y_test= [ones(sum(testinc),1); zeros(sum(testnorm),1)];
    % weight LDA according to number of samples
    weights = [ones(sum(~testinc),1)*size(X_train,1)/sum(~testinc);...
        ones(sum(~testnorm),1)*size(X_train,1)/sum(~testnorm);];
    clf_lda = fitcdiscr(X_train,Y_train,'weights',weights);
    predictResult = predict(clf_lda,X_test);
    predTrain = predict(clf_lda,X_train);
    
    switch vali_method
        case 'fold'
            inc_acc(f_i) = sum(predictResult&Y_test)/sum(Y_test);
            norm_acc(f_i) = sum(~predictResult&~Y_test)/sum(~Y_test);
            inc_acc_tr(f_i) = sum(predTrain&Y_train)/sum(Y_train);
            norm_acc_tr(f_i) = sum(~predTrain&~Y_train)/sum(~Y_train);
            balanced_acc(f_i) = (inc_acc(f_i)+norm_acc(f_i))/2;
            pos_pred_val(f_i) = sum(predictResult&Y_test)/sum(predictResult);
            neg_pred_val(f_i) = sum(~predictResult&~Y_test)/sum(~predictResult);
%         case 'looSess'
        otherwise
            pred(f_i) = predictResult;
            inc_acc_tr(f_i) = sum(predTrain&Y_train)/sum(Y_train);
            norm_acc_tr(f_i) = sum(~predTrain&~Y_train)/sum(~Y_train);
    end

end

if ~strcmp(vali_method,'fold')
    pred = logical(pred);
    tmp_inc = pred(1:length(label_increase));
    tmp_norm = pred(length(label_increase)+1:end);
    inc_acc = sum(tmp_inc)/length(label_increase);
    norm_acc = sum(~tmp_norm)/length(label_normal);
    balanced_acc = (inc_acc+norm_acc)/2;
    pos_pred_val = sum(tmp_inc)/sum(pred);
    neg_pred_val = sum(~tmp_norm)/sum(~pred);
    mis_classified_inc = label_increase(~tmp_inc);
    mis_classified_norm = label_normal(tmp_norm);
    mis_classified_file = [mis_classified_inc; mis_classified_norm];
    varargout{1} = mis_classified_file;
end


disp('-------------------')
fprintf('Avg. Test Balanced ACC = %2.1f%%\n',mean(balanced_acc)*100);
disp('-------------------')
fprintf('Avg. Test INC ACC = %2.1f%%\n',mean(inc_acc)*100);
disp('-------------------')
fprintf('Avg. Test NORM ACC = %2.1f%%\n',mean(norm_acc)*100);
disp('-------------------')
fprintf('Avg. Pos. Pred. Val.= %2.1f%%\n',mean(pos_pred_val)*100);
disp('-------------------')
fprintf('Avg. Neg. Pred. Val.= %2.1f%%\n',mean(neg_pred_val)*100);
disp('-------------------')
fprintf('Avg. Train INC ACC = %2.1f%%\n',mean(inc_acc_tr)*100);
disp('-------------------')
fprintf('Avg. Train NORM ACC = %2.1f%%\n',mean(norm_acc_tr)*100);
disp('-------------------')




end

function [test_inc_lib, test_norm_lib] = split_fold(nfold,label_increase,label_normal)
%% generate Fold data
step_inc = ceil(length(label_increase)/nfold);
step_norm = ceil(length(label_normal)/nfold);
% shuffle
shuffled_label_inc = label_increase(randperm(length(label_increase)));
shuffled_label_norm = label_normal(randperm(length(label_normal)));
test_inc_lib = cell(1,nfold);
test_norm_lib = cell(1,nfold);

for fold_idx = 1:nfold
    if fold_idx*step_inc <= length(label_increase)
        test_inc_idx = shuffled_label_inc((fold_idx-1)*step_inc+1:fold_idx*step_inc);
        test_norm_idx = shuffled_label_norm((fold_idx-1)*step_norm+1:fold_idx*step_norm);
    else
        test_inc_idx = shuffled_label_inc(end-step_inc+1:end);
        test_norm_idx = shuffled_label_norm(end-step_norm+1:end);
    end
    test_inc_lib{fold_idx} = test_inc_idx;
    test_norm_lib{fold_idx} = test_norm_idx;
end

end












