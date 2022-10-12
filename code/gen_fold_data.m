function gen_fold_data(icawm_lib_inc, icawm_lib_norm, inc_list_lib, norm_list_lib, test_inc_idx, test_norm_idx, label_increase, label_normal, chan30, fold_idx)
filepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/ASR/ICrej/';
%% creating spatial filter
icawm_lib_avg = zeros(30,5); %channel by cluster
icawm_lib_med = zeros(30,5); %channel by cluster

for cls_i = 1:5
    temp_inc = icawm_lib_inc{cls_i};
    temp_norm = icawm_lib_norm{cls_i};
    inc_wm = cell2mat(temp_inc(~ismember(inc_list_lib{cls_i},test_inc_idx)));
    norm_wm = cell2mat(temp_norm(~ismember(norm_list_lib{cls_i},test_norm_idx)));
    icawm_lib_avg(:,cls_i) = mean([inc_wm,norm_wm],2);
    icawm_lib_med(:,cls_i) = median([inc_wm,norm_wm],2);
end

%% calculate projected IC PSD
plt_avgall_inc = zeros(length(label_increase),50,5); % subj by freq by cluster
plt_medall_inc = zeros(length(label_increase),50,5);
plt_avgall_norm = zeros(length(label_normal),50,5);
plt_medall_norm = zeros(length(label_normal),50,5);

parfor i = 1:length(label_increase)
    subj_i = label_increase(i);
    EEG = pop_loadset([filepath,sprintf('%d.set',subj_i)]);
    ch_idx = ismember(chan30,{EEG.chanlocs.labels});
    tmpdata = zeros(10,EEG.pnts);
    for cls_i = 1:5
        tmpdata(cls_i*2-1,:) = icawm_lib_avg(ch_idx,cls_i)'*EEG.data;
        tmpdata(cls_i*2,:) = icawm_lib_med(ch_idx,cls_i)'*EEG.data;
    end
    [spectra,~] = spectopo(tmpdata,0,EEG.srate,'plot','off');
    for cls_i = 1:5
        plt_avgall_inc(i,:,cls_i) = spectra(cls_i*2-1,1:50);
        plt_medall_inc(i,:,cls_i) = spectra(cls_i*2,1:50);
    end
end


parfor i = 1:length(label_normal)
    subj_i = label_normal(i);
    EEG = pop_loadset([filepath,sprintf('%d.set',subj_i)]);
    ch_idx = ismember(chan30,{EEG.chanlocs.labels});
    tmpdata = zeros(10,EEG.pnts);
    for cls_i = 1:5
        tmpdata(cls_i*2-1,:) = icawm_lib_avg(ch_idx,cls_i)'*EEG.data;
        tmpdata(cls_i*2,:) = icawm_lib_med(ch_idx,cls_i)'*EEG.data;
    end
    [spectra,~] = spectopo(tmpdata,0,EEG.srate,'plot','off');
    for cls_i = 1:5
        plt_avgall_norm(i,:,cls_i) = spectra(cls_i*2-1,1:50);
        plt_medall_norm(i,:,cls_i) = spectra(cls_i*2,1:50);
    end
end


com = sprintf('Leave-one-session out. All 30 channels. Fold %d.',fold_idx);
savepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/ASR/ICrej/cls_idx_k5/fold_data/';
savename = sprintf('icawm_allCh_fold%d.mat',fold_idx);
save([savepath,savename],'plt_avgall_norm','plt_medall_norm','plt_avgall_inc','plt_medall_inc',...
    'label_increase','label_normal','icawm_lib_avg','icawm_lib_med',...
    'test_inc_idx','test_norm_idx','com');
disp('Done')


end