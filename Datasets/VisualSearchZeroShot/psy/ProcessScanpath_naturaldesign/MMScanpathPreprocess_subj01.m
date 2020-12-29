clear all; close all; clc;

%% We are using EYELINK II; sampling rate: 500Hz
%% We define 50Hz as fixation duration (20 consecutives per group)
%% monitor resolution: window_rect = [0 0 1280 1024] for eyetracking
addpath('../');
taskdirectory = 'subjects_naturaldesign';
subjname = 'subj01-mm';
subjname = 'subj02-az';
subjname = 'subj03-el';
subjname = 'subj04-ni';
subjname = 'subj05-mi';
subjname = 'subj06-st';
subjname = 'subj07-pl';
subjname = 'subj08-su';
subjname = 'subj09-an';
subjname = 'subj10-ni';
subjname = 'subj11-ta';
subjname = 'subj12-mi';
subjname = 'subj13-zw';
subjname = 'subj14-ji';
subjname = 'subj15-ra';
subjname = 'subj16-kr';
subjname = 'subj17-ke';

edflist = dir(['../' taskdirectory '/' subjname '/CVS_MM_' subjname '-*.edf']);

load('../../SubjectArray/naturaldesign.mat');
load('../../SubjectArray/naturaldesign_seq.mat');
NumTrials = length(MyData);
trial = 1;
MaxFixNum = 80;
Fix_posx = {};
Fix_posy = {};
Fix_time = {};
Fix_starttime = {};
Mouseclicktime = {};
TargetFound = [];
FixData = [];
screenx = 1280; %monitor screen size
screeny = 1024;
maxfixlength = 0;
%2,5
%% Converting the EDF File and saving it as a Matlab File

subjclickwrong = nan(NumTrials,1);

for i = 1:length(edflist)
    i
    
%     if i == 2
%         continue;
%     end
    
    edf0 = Edf2Mat(['../' taskdirectory '/' subjname '/' edflist(i).name]);
    eventlist = edf0.RawEdf.FEVENT;
    Time = edf0.Samples.time;
    ToSearchString = ['ENDFIX'];
    fixindex=structfind(eventlist,'codestring',ToSearchString);
    
    ToSearchString = ['TARGET FOUND'];
    targetindex = structfind(eventlist,'message',ToSearchString);
    
    ToSearchString = ['WRONG MOUSE CLICK'];
    clickwrongindex = structfind(eventlist, 'message', ToSearchString);
    
    while trial <= NumTrials
        trial
        ToSearchString = ['TRIAL_ON: ' num2str(trial) ];
        startindex=structfind(eventlist,'message',ToSearchString);
        ToSearchString = ['TRIAL_OFF: ' num2str(trial)];
        endindex=structfind(eventlist,'message',ToSearchString);
        
        if isempty(startindex) || isempty(endindex)
            break;
        end
        
        %process fixations       
        filteredfixindex = fixindex(find(fixindex >= startindex & fixindex <= endindex+1)); %include the last fixation just before trial ends
        fixx = [];
        fixy = [];
        fixtime = [];
        fixstarttime = [];
        for j = 1: length(filteredfixindex)
            fixx = [fixx eventlist(filteredfixindex(j)).gavx];
            fixy = [fixy eventlist(filteredfixindex(j)).gavy];
            
            if j ==1
                time = eventlist(filteredfixindex(j)).entime - eventlist(startindex).sttime;
                starttime = eventlist(filteredfixindex(j)).sttime - eventlist(startindex).sttime;
            else
                time = eventlist(filteredfixindex(j)).entime - eventlist(startindex).sttime;
                starttime = eventlist(filteredfixindex(j)).sttime - eventlist(startindex).sttime;
            end
            fixtime = [fixtime time];
            fixstarttime = [fixstarttime starttime];
        end
        Fix_posx = [Fix_posx; int32(fixx)];
        Fix_posy = [Fix_posy; int32(fixy)];
        Fix_time = [Fix_time; fixtime];

        Fix_starttime = [Fix_starttime; fixstarttime];

        %process target flag
        filteredtargetflag = find(targetindex >=startindex & targetindex <= endindex);
        
        fixtargetseq = zeros(1,MaxFixNum);
        
        %process wrong click flag
        filteredwrongclickflag = find(clickwrongindex >=startindex & clickwrongindex <= endindex);
        
        if isempty(filteredtargetflag)
            TargetFound = [TargetFound; zeros(1,MaxFixNum)];
            Mouseclicktime = [Mouseclicktime; nan];
        else
            Mouseclicktime = [Mouseclicktime; eventlist(targetindex(filteredtargetflag)).sttime - eventlist(startindex).sttime];
            if isempty(filteredwrongclickflag)
                subjclickwrong(trial) = 0;
            else
                subjclickwrong(trial) = 1; %yes; wrong click before finding target
            end
            
            fixRight = length(find(filteredfixindex < targetindex(filteredtargetflag))) ;
            if fixRight <= 1
                fixRight = 1;     
            end
            fixtargetseq(1: fixRight-1) = 0;
            fixtargetseq(  fixRight ) = 1;
            TargetFound = [TargetFound; fixtargetseq];
        end
        
        %to check fixations; plot them out
        img = imread(['../../../Datasets/NaturalDesign/img' sprintf('%03d',MyData(seq(trial))) '.jpg']);
        img = imresize(img,[screeny screenx]);
        
        if maxfixlength < length(filteredfixindex)
            maxfixlength = length(filteredfixindex);
        end
        
        
%          fixnumstr = cellstr(string([1:1:length(fixx)]));
%          RGB = insertText(rgb2gray(img),[int32(fixx(1:end)); int32(fixy(1:end))]', fixnumstr);
%          imshow(RGB);
%          pause;
        

        trial = trial + 1;  
    end
end

FixData.Fix_posx = Fix_posx;
FixData.Fix_posy = Fix_posy;
FixData.TargetFound = TargetFound(:,1:MaxFixNum);
FixData.Fix_time = Fix_time;
FixData.Fix_starttime = Fix_starttime;
FixData.Mouseclicktime = Mouseclicktime;
save([subjname '.mat'],'FixData','subjclickwrong');
plot(cumsum(nanmean(TargetFound(:,2:end,1))))

nanmean(subjclickwrong)







