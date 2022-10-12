%% Project data onto component space using template ICA weights * ICA sphere
asrpath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/ASR/';
projpath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/ASR/Projected_rmEye/';
icapath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/ASR/ICrej/';
rawpath = '/home/yuan/Documents/2019 stress in classroom/resting collection/raw/';
EEG = pop_loadset('1.set');
template_icawm = EEG.icaweights*EEG.icasphere;
template_icawinv = EEG.icawinv;
template_icawinv(:,1:2) = 0;

%%
% f_list = [1:58,60:81,83:89,91,92,94:110];
f_list = [label_increase',label_normal'];
ch_error = false(1,length(f_list));
parfor j = 1:length(f_list)
    i = f_list(j);
    filename = sprintf('%d.set',i);
    if exist([asrpath, filename],'file') && exist([icapath, filename],'file') && ~exist([projpath,filename],'file')
        EEG = pop_loadset([asrpath, filename]);
        EEG_ica = pop_loadset([icapath, filename]);
        if EEG.nbchan ~= EEG_ica.nbchan || length({EEG.chanlocs.labels})~=length({EEG_ica.chanlocs.labels})
            ch_error(j) = true;
        end
        nbchan = EEG.nbchan;
        tar_ch_idx = ismember(chan30, {EEG.chanlocs.labels});
        fprintf('>> Enter: %d\n',i);
        % recording ch by comp by comp by ch
        proj_mat = template_icawinv(tar_ch_idx,1:nbchan)*template_icawm(1:nbchan,tar_ch_idx);
        EEG.data = proj_mat*EEG.data;
        pop_saveset(EEG,[projpath, filename]);
        fprintf('\nfinished %d\n',i);
    end
end

disp('Done')

%% put ICA information back to file
% f_list = [1:46,48:58,60:81,83:89,91,92,94:110];
% 
% parfor j = 1:length(f_list)
%     i = f_list(j);
%     filename = sprintf('%d.set',i);
%     EEG_ica = pop_loadset([icapath,filename]);
%     EEG_asr = pop_loadset([asrpath,filename]);
%     EEG_asr.icaweights = EEG_ica.etc.icaweights_beforerms;
%     EEG_asr.icasphere = EEG_ica.etc.icasphere_beforerms;
%     EEG_asr.icawinv = pinv(EEG_asr.icaweights*EEG_asr.icasphere);
%     pop_saveset(EEG_asr,filename,asrpath);
% end
% disp('Done')

