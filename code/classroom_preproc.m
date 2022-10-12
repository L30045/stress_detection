%% extract resting data
%% file location and EEG channel labels
filepath = 'E:\NCTU_RWN-SFC\EEG+ECG\';
file_list = readtable([filepath(1:end-8), 'DSS+DASS21/DASS21/SFC_DASS21.xlsx']);
filepath_lib = cellfun(@(x) x(2:end-1),file_list.session,'uniformoutput',0);
subj_list = file_list.subject;
summary_list = readtable([filepath,'summary_NCTU_RWN-SFC.xls']);
summary_subj = lower(summary_list.Var5);
% tar_ch = {'C3','C4','CP3','CP4','CPZ','CZ','F3','F4','F7','F8','FC3','FC4',...
%     'FCZ','FP1','FP2','FT7','FT8','FZ','O1','O2','OZ','P3','P4','PZ','T3',...
%     'T4','T5','T6','TP7','TP8','P7','P8','T7','T8'};
savepath = 'E:\NCTU_RWN-SFC\EEG+ECG\resting collection\';
error_EEG = [];
rmCh_lib = cell(size(subj_list));

%% channl location
chan_NuAmps = readtable([filepath, '30ch_loc_NuAmps.xls']);
chan_SynAmps = readtable([filepath, '30ch_loc_SynAmps2.xls']);

%% EEG preprocessing
for i = [120,121,]
    fprintf('>>>>>>> subj %d\n\n', i);
    switch file_list.semester(i)
        case 1
            loc_path = [filepath, 'first semester\', filepath_lib{i}];
            loc_f = [loc_path,filesep,sprintf('s%d.set',subj_list(i))];
        case 2
            loc_path = [filepath, 'second semester\', filepath_lib{i}];
            loc_f = [loc_path,filesep,sprintf('%s_S%d_ev.set',filepath_lib{i}(1:8),subj_list(i)-18)];
    end
    savename = sprintf('%d.set',i);
    % channl location
    tar_idx = ismember(summary_list.Var4, filepath_lib{i}) & ismember(summary_subj, sprintf('s%02d',subj_list(i)));
    chan_path = summary_list.Var17(tar_idx);
    switch chan_path{:}
        case '30ch_loc_NuAmps.xls'
            chan_lib = chan_NuAmps;
        case '30ch_loc_SynAmps2.xls'
            chan_lib = chan_SynAmps;
    end
    tar_ch = cellfun(@(x) x(2:end-1),chan_lib.label,'uniformoutput',0);
    % load EEG
    if i == 153
        loc_f = [loc_path,filesep,'20150506_S6.cnt'];
        EEG = pop_loadcnt(loc_f);
        rest_start = 23761;
        rest_end = 323761;
    else
        EEG = pop_loadset(loc_f);
        % extract resting section
        rest_start = EEG.event(1).latency;
        rest_end = EEG.event(2).latency;
    end
    if (rest_end - rest_start)/EEG.srate/60 < 1
        error_EEG = [error_EEG; i];
    else
        EEG = pop_select(EEG,'point',[rest_start, rest_end]);
        % preprocessing
        % bandpass
        EEG = pop_eegfiltnew(EEG,0.5,50);
        % remove irrelevant chs
        preserveCh = ismember({EEG.chanlocs.labels},tar_ch);
        EEG = pop_select(EEG, 'channel', {EEG.chanlocs(preserveCh).labels});
        % channel location lookup
        for ch_i = 1:length(EEG.chanlocs)
            t_i = ismember(tar_ch,EEG.chanlocs(ch_i).labels);
            EEG.chanlocs(ch_i).theta = chan_lib.theta(t_i);
            EEG.chanlocs(ch_i).radius= chan_lib.radius(t_i);
            EEG.chanlocs(ch_i).X = chan_lib.X(t_i);
            EEG.chanlocs(ch_i).Y = chan_lib.Y(t_i);
            EEG.chanlocs(ch_i).Z = chan_lib.Z(t_i);
            EEG.chanlocs(ch_i).sph_theta = chan_lib.sph_theta(t_i);
            EEG.chanlocs(ch_i).sph_phi = chan_lib.sph_phi(t_i);
            EEG.chanlocs(ch_i).sph_radius = chan_lib.sph_radius(t_i);
        end
        % reref
        ref_ch = [];
        EEG = pop_reref(EEG, ref_ch);
        % bad channel removal + ASR
        EEG = clean_rawdata(EEG,5,-1,0.7,4,20,-1);
        rmCh_lib{i} = setdiff(tar_ch,{EEG.chanlocs.labels});
        % save resting data
        pop_saveset(EEG,savename,savepath);
    end
end
% save([savepath,'errorEEG.txt'], 'error_EEG', '-ASCII');
% save([savepath,'rmCh_lib.mat'], 'rmCh_lib');
disp('Done')

%%
for i = 1:171
    if ~ismember(i,error_EEG)
        EEG = pop_loadset([savepath,sprintf('%d.set',i)]);
        data = EEG.data;
        save([savepath,sprintf('preprocessed/%d.mat',i)],'data');
    end
end