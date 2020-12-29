clear all; close all; clc;

%% Author: Mengmi Zhang
%% Kreiman Lab
%% web: http://klab.tch.harvard.edu/
%% Date: April 5, 2018

subjnum = 2; %1-20 subjects
load(['naturaldesign.mat']);
load(['naturaldesign_seq.mat']);
MyData = MyData(seq);
NumTrial = length(MyData);
colormode = 'color'; %'color','grey'
ImageDir = ['../stimuli/'];

global params;
params = struct();

commandwindow;

results={};
c=clock;

params.NBTRIALSPERBLOCK = 10;
params.NBTRIALSBETWEENRNGRESETS = 30;

params.NBOBJS = 40;
params.ARRAYSIZE=6;
params.PROBATARGETAMONGPROBES = .75;
params.NBTRIES = 10;
params.ECCENTRICITY = 300; % 400;
params.SHINED=0; % 0
params.CSIZE=156; % 156

KbName('UnifyKeyNames');

params.session_start_time=c;
params.monitor_ID = 0; % 1 = additional monitor, 0 = built-in
params.fixation_threshold=1.0;% radius in visual angle degrees around fixation point to count as fixation
params.fixation_time=0.5;% time in seconds fixation must be maintained before trial starts
params.fixation_timeout=10;% number of seconds to wait at fixation point before asking about recalibration
params.exit_key=KbName('Escape');
params.eyelink_file = ['testmengmi.edf']; %this is supposed to change
params.font_size=24;
params.text_color=0;

params.STARTTRIALFIXATIONDELAY = 1.0;
params.ARRAYPRESENTATIONTIME = .3;
params.TARGETPRESENTATIONTIME = 2.0;
params.TARGETARRAYDELAY = 1.0;
params.MAXTRIALTIME = 2.0;

Screen('Screens');	% make sure all functions (SCREEN.mex) are in memory
HideCursor;	                                % Hides the mouse cursor
FlushEvents('keyDown');

% open psychophysics display
[window,window_rect]=Screen('OpenWindow',params.monitor_ID);
[params.screenXpixels, params.screenYpixels] = Screen('WindowSize', window);

params.window = window;
params.window_rect = window_rect;
params.ctrx = floor(window_rect(3)/2);
params.ctry = floor(window_rect(4)/2);
params.white = WhiteIndex(window); % pixel value for white
params.black = BlackIndex(window); % pixel value for black
params.gray = floor((params.white+params.black)/2);

params.WaitSearch = 20; %wait for 20 seconds to go to the next trial if target not found
priorityLevel=MaxPriority(window);% Set priority for script execution to realtime priority
Priority(priorityLevel);
Screen('TextSize',window,params.font_size);

try  % Put it in a try-catch, so if an error happens, you don't have to deal with a stuck screen...
   
    line1 = 'Loading -';
    line2 = '\n When screen goes blank,';
    line3 = '\n press "c" to start calibration';
    DrawFormattedText(window, [line1 line2 line3],'center', params.screenYpixels * 0.45, params.black);   
    Screen('flip',window);
    WaitSecs(0.2);
    KbWait();
   
   
    line1='Ready.';
    line2='\n Press any key to begin, or move to next trial.';
    line3 ='\n Press ESCAPE repeatedly to exit program.';   
    DrawFormattedText(window, [line1 line2 line3],'center', params.screenYpixels * 0.45, params.black);   
    Screen('Flip',window);    
    WaitSecs(0.2); 
    KbWait();

    %% lets start the trials
    for t = 1: NumTrial
        
        trial = MyData(t);
    
        % draw cross on the screen
        %            Screen('fillrect',params.window,  params.gray);
        Screen('fillrect',params.window,  params.black, [params.ctrx-14 params.ctry-1 params.ctrx+14 params.ctry+1]);
        Screen('fillrect',params.window,  params.black, [params.ctrx-1 params.ctry-14 params.ctrx+1 params.ctry+14]);
        Screen('Flip', window);      
        WaitSecs(params.STARTTRIALFIXATIONDELAY);

        % Clear screen
        Screen('FillRect', window,  params.gray);
        Screen('Flip', window);
        WaitSecs(0.2);
        
        img = imread([ImageDir MyData(t).targetname]);
        img = imresize(img, [window_rect(4) window_rect(3)]);
        gt = imread([ImageDir '../gt/gt' MyData(t).targetname(8:end)]);
        gt = double(imresize(gt, [window_rect(4) window_rect(3)])/255);
        gt = mat2gray(im2bw(gt,0.5));
        %max(max(gt))
        textures = Screen('MakeTexture', window, img);
        Screen('DrawTexture', window, textures, [], window_rect);        
        onsettime = Screen(window, 'Flip');
        
        %% This is the part for checking mouse clicks and waiting for seconds
        t_start = GetSecs();
        stopflag = 0;        
        while stopflag == 0            
            [x,y,buttons] = GetMouse;            
            t_now = GetSecs();
            
            if gt(y,x) == 1 && buttons(1) == 1
                display('yes, target found!');
                stopflag = 1;
                break;
            end
            
            if (t_now - t_start) > params.WaitSearch
                display('time exceed');
                stopflag = 1;
                break;
            end
                        
            [keyIsDown, secs, keyCode] = KbCheck;
             if keyCode(params.exit_key)
                display(['ESC pressed!']);           
                break;
             end
            
        end        
                
        %WaitSecs(params.ARRAYPRESENTATIONTIME);
        %KbWait;
        if stopflag == 0
            display(['ESC pressed!']);  
            break;
        end
        

         %WaitSecs(.2); %lets start a new trial

      
   end % main loop   
   WaitSecs(0.1);

   %draw_centered_text(window,{'Thanks!'},params.text_color);
   line1 ='Thanks!';   
   DrawFormattedText(window, [line1],'center', params.screenYpixels * 0.5, params.black);
   Screen('flip',window);
   WaitSecs(.5);
   Screen('CloseAll');
   
   sca;
catch % An error has occurred !
    
   Screen('CloseAll');
   err = lasterror; disp(err.message); for stack_ind = 1:length(err.stack); disp(err.stack(stack_ind));  end;

end
% *** End main function *** %
