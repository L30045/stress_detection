% path setting
filepath = 'E:\NCTU_RWN-SFC\EEG+ECG\';
file_list = readtable('E:\NCTU_RWN-SFC\DSS+DASS21\DASS21\SFC_DASS21_before stress.xlsx');
filepath_lib = file_list.Var1;
for i = 1:length(filepath_lib)
    if filepath_lib{i}(1)==''''
        filepath_lib{i} = filepath_lib{i}(2:end-1);
    else
        filepath_lib{i} = filepath_lib{i}(1:end-1);
    end
end
subj_list = file_list.Var2;
summary_list = readtable([filepath,'summary_NCTU_RWN-SFC.xls']);
summary_subj = lower(summary_list.Var5);

savepath = 'E:\NCTU_RWN-SFC\EEG+ECG\epoch\';
if ~exist(savepath,'dir')
    mkdir(savepath)
end
error_EEG = [];
rmCh_lib = cell(size(subj_list));

%% channl location
chan_NuAmps = readtable([filepath, '30ch_loc_NuAmps.xls']);
chan_SynAmps = readtable([filepath, '30ch_loc_SynAmps2.xls']);

%% process
for i = 1:length(filepath_lib)
    fprintf('>>>>>>> subj %d\n\n', i);
    loc_path = [filepath, 'second semester\', filepath_lib{i},'\'];
    file_lib = dir(loc_path);
    file_lib = {file_lib.name};
    f_id = file_lib{~cellfun(@isempty,(regexp(file_lib,sprintf('S%d(?:\\w*).set',subj_list(i)-18),'ONCE')))};
	loc_f = [loc_path,f_id];
    savename = [f_id(1:9),sprintf('s%d',subj_list(i)),f_id(15:end)];
    
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
    EEG = pop_loadset(loc_f);
    % extract resting section
    if isa(EEG.event(1).type,'double')
        rest_b_start = EEG.event(find([EEG.event.type]==119)).latency; 
        exam_start = EEG.event(find([EEG.event.type]==105)).latency; 
        rest_a_start = EEG.event(find([EEG.event.type]==219)).latency;
        question_flag = any([EEG.event.type]==110);
        eeg_ev = [EEG.event.type];
    elseif isa(EEG.event(1).type,'char')
        rest_b_start = EEG.event(ismember({EEG.event.type},'119')).latency; 
        exam_start = EEG.event(ismember({EEG.event.type},'105')).latency; 
        rest_a_start = EEG.event(ismember({EEG.event.type},'219')).latency; 
        question_flag = any(cellfun(@str2double, {EEG.event.type})==110);
        eeg_ev = cellfun(@str2double, {EEG.event.type});
    else
        error('Missing event marker.')
    end
    % sanity check
%     figure
%     plt_ev = cellfun(@str2double, {EEG.event.type});
%     plot(plt_ev)
    
    % calculate resting duration
    rest_duration = exam_start-rest_b_start;
    if rest_duration/EEG.srate/60 < 1 % min
        error_EEG = [error_EEG; i];
    end
    % compare the resting duration with the next event
    if find(eeg_ev==219)==length(eeg_ev)
        rest_a_end = rest_a_start+rest_duration;
    else
        rest_a_end = min([EEG.event(find(eeg_ev==219)+1).latency...
                      rest_a_start+rest_duration]);
    end
    % extract data    
    if question_flag
        % preprocessing
        % remove irrelevant chs
        preserveCh = ismember({EEG.chanlocs.labels},tar_ch);
        EEG = pop_select(EEG, 'channel', {EEG.chanlocs(preserveCh).labels});
        % reref
%         ref_ch = summary_list.NA_6(tar_idx);
%         if ~isempty(ref_ch{:})
%             ref_ch = regexp(ref_ch,'[^\w]','split');
%             ref_ch = ref_ch{:};
%             ref_ch = ref_ch(~cellfun(@isempty, ref_ch));
%             EEG = pop_select(EEG, 'nochannel',ref_ch);
%         end
        EEG = pop_reref(EEG, []);
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
        % bandpass
        EEG = pop_eegfiltnew(EEG,1,50);
        % bad channel removal + ASR
        EEG = clean_rawdata(EEG,5,-1,0.7,4,20,-1);
        % epoch question section
        q_start = [EEG.event(eeg_ev==110).latency];
        q_end = [EEG.event(eeg_ev==120).latency];
        EEG = pop_select(EEG,'point',[rest_b_start, exam_start; q_start', q_end'; rest_a_start, rest_a_end]);
    else
        EEG = pop_select(EEG,'point',[rest_b_start, rest_a_end]);
        % preprocessing
        % remove irrelevant chs
        preserveCh = ismember({EEG.chanlocs.labels},tar_ch);
        EEG = pop_select(EEG, 'channel', {EEG.chanlocs(preserveCh).labels});
        % reref
%         ref_ch = summary_list.NA_6(tar_idx);
%         if ~isempty(ref_ch{:})
%             ref_ch = regexp(ref_ch,'[^\w]','split');
%             ref_ch = ref_ch{:};
%             ref_ch = ref_ch(~cellfun(@isempty, ref_ch));
%             EEG = pop_select(EEG,'nochannel',ref_ch);
%         end
        EEG = pop_reref(EEG, []);
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
        % bandpass
        EEG = pop_eegfiltnew(EEG,1,50);
        % bad channel removal + ASR
        EEG = clean_rawdata(EEG,5,-1,0.7,4,20,-1);
    end
    rmCh_lib{i} = setdiff(tar_ch,{EEG.chanlocs.labels});
    % save resting data
    pop_saveset(EEG,savename,savepath);
end
save([savepath,'errorEEG.txt'], 'error_EEG', '-ASCII');
save([savepath,'rmCh_lib.mat'], 'rmCh_lib');
disp('Done')