function gen_looSess_data(icawm_lib_inc, icawm_lib_norm, inc_list_lib, norm_list_lib, label_increase, label_normal, test_sess, chan30, varargin)
filepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/ASR/ICrej/';
%% creating spatial filter
icawm_lib_avg = zeros(30,5); %channel by cluster
icawm_lib_med = zeros(30,5); %channel by cluster

for cls_i = 1:5
    temp_inc = icawm_lib_inc{cls_i};
    temp_norm = icawm_lib_norm{cls_i};
    inc_wm = cell2mat(temp_inc(~ismember(inc_list_lib{cls_i},test_sess)));
    norm_wm = cell2mat(temp_norm(~ismember(norm_list_lib{cls_i},test_sess)));
    icawm_lib_avg(:,cls_i) = mean([inc_wm,norm_wm],2);
    icawm_lib_med(:,cls_i) = median([inc_wm,norm_wm],2);
end

%% calculate projected IC PSD
plt_avgall_inc = zeros(length(label_increase),50,5); % subj by freq by cluster
plt_medall_inc = zeros(length(label_increase),50,5);
plt_avgall_norm = zeros(length(label_normal),50,5);
plt_medall_norm = zeros(length(label_normal),50,5);


if isempty(varargin)
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
    com = sprintf('Leave-one-session out. All 30 channels. Test %d.',test_sess);
    savepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/ASR/ICrej/cls_idx_k5/LOO_sess/';
    savename = sprintf('icawm_allCh_looSess%d.mat',test_sess);
    save([savepath,savename],'plt_avgall_norm','plt_medall_norm','plt_avgall_inc','plt_medall_inc',...
        'label_increase','label_normal','icawm_lib_avg','icawm_lib_med',...
        'test_sess','com');
    disp('Done')
else
    tarCh = varargin{1};
    icawm_lib_avg = icawm_lib_avg(ismember(chan30,tarCh),:);
    icawm_lib_med = icawm_lib_med(ismember(chan30,tarCh),:);
    parfor i = 1:length(label_increase)
        subj_i = label_increase(i);
        EEG = pop_loadset([filepath,sprintf('%d.set',subj_i)]);
        ch_idx = ismember({EEG.chanlocs.labels},tarCh);
        tmpdata = zeros(10,EEG.pnts);
        for cls_i = 1:5
            tmpdata(cls_i*2-1,:) = icawm_lib_avg(:,cls_i)'*EEG.data(ch_idx,:);
            tmpdata(cls_i*2,:) = icawm_lib_med(:,cls_i)'*EEG.data(ch_idx,:);
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
        ch_idx = ismember({EEG.chanlocs.labels},tarCh);
        tmpdata = zeros(10,EEG.pnts);
        for cls_i = 1:5
            tmpdata(cls_i*2-1,:) = icawm_lib_avg(:,cls_i)'*EEG.data(ch_idx,:);
            tmpdata(cls_i*2,:) = icawm_lib_med(:,cls_i)'*EEG.data(ch_idx,:);
        end
        [spectra,~] = spectopo(tmpdata,0,EEG.srate,'plot','off');
        for cls_i = 1:5
            plt_avgall_norm(i,:,cls_i) = spectra(cls_i*2-1,1:50);
            plt_medall_norm(i,:,cls_i) = spectra(cls_i*2,1:50);
        end
    end
    com = sprintf('Leave-one-session out. FZ,FCZ,CZ channels. Test %d.',test_sess);
    savepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/ASR/ICrej/cls_idx_k5/LOO_sess_proj/';
    savename = sprintf('icawm_proj_looSess%d.mat',test_sess);
    save([savepath,savename],'plt_avgall_norm','plt_medall_norm','plt_avgall_inc','plt_medall_inc',...
        'label_increase','label_normal','icawm_lib_avg','icawm_lib_med',...
        'test_sess','com');
    disp('Done')
end

end