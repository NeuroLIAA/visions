function d = fun_extract_samples(todo)


%%
todo.samples(:,5) = zeros(length(todo.samples),1);

N = length(todo.msgtime);

for i=1:N
    
    switch todo.msg{i}
        
        case 'Sync PreCue ON'
            
            ind = find( todo.msgtime(i) == todo.samples(:,1) );
            
            todo.samples(ind, 5) = 1;
            
        case 'Sync BlankScreen ON'
            
            ind = find( todo.msgtime(i) == todo.samples(:,1) );
            
            todo.samples(ind, 5) = 2;
    
        case 'Sync Stimulus ON'
            
            ind = find( todo.msgtime(i) == todo.samples(:,1) );
            
            todo.samples(ind, 5) = 3;
    
        case 'Sync Stimulus OFF'
            
            [a ind] = min( abs(todo.msgtime(i) - todo.samples(:,1)) );
            
            todo.samples(ind, 5) = 4;
    
    end
    
end

%%

ind_start = find(todo.samples(:,5)==1);
t_bs      = find(todo.samples(:,5)==2) - ind_start + 1;
t_stim    = find(todo.samples(:,5)==3) - ind_start + 1;
ind_end   = find(todo.samples(:,5)==4);

Ntrials   = sum(todo.samples(:,5)==1);

for trial=1:Ntrials    
    
    ftrial = ind_start(trial):ind_end(trial);
    
    d(trial).msg                = zeros(length(ftrial), 1);
    d(trial).msg(t_bs(trial))   = 1;
    d(trial).msg(t_stim(trial)) = 2;
    
    d(trial).xR          = todo.samples(ftrial,2);
    d(trial).yR          = todo.samples(ftrial,3);
    d(trial).pupilR      = todo.samples(ftrial,4);
    
end
    
    
    

%%

    