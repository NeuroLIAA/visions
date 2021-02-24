clear all
close all
clc

M       = 0;

addpath('../dynamic_models/utils/')

dirdata = '~/Dropbox/10_datos_pesados/vs_models_data/';

dirtodo = [dirdata 'todo/'];
dirmat  = [dirdata 'mat/'];
dirin   = [dirdata 'info/'];
dirsmp  = [dirdata 'sinfo/'];

% dir_images      = '/home/usuario/repos/vs_models/images';
% dir_templates   = '/home/usuario/repos/vs_models/templates';
dir_images      = '/home/juank/repo/vs_models/images';
dir_templates   = '/home/juank/repo/vs_models/templates';

% create_info_from_raw_data_new('../../../data/vs_models/all_data/')
% system(['cp ../../../data/vs_models/all_data/*_info.mat ' dirin])

tmp=dir([dirmat '*.mat']); tmp = {tmp.name};
subjnames = cellfun(@(x) x(1:end-4),tmp,'UniformOutput',0);

su = 1
    fprintf('**********************************************************\n')
    fprintf('%s \n',subjnames{su})
    
    matfile = [subjnames{su} '.mat'];
    filein  = [subjnames{su} '_info.mat'];
    fileout = [subjnames{su} '_simple.mat'];
%     load([dirmat matfile]);
%     load([dirin filein]);
    load([dirsmp fileout]);

    for tr = 1:length(smpinfo)
        % mappedFix = mapToReducedMatrix(fixation, delta, image_size)
        mappedFix = mapToReducedMatrix([smpinfo(tr).x' smpinfo(tr).y'], 32, smpinfo(tr).image_size);
        smpinfo(tr).x_grid = mappedFix(:,1)';
        smpinfo(tr).y_grid = mappedFix(:,2)'; 
    end

%% Check: Overall fixation positions
if 1
addpath('../data_analysis/utils/heatmap_code');
settings.filterSize = 300; % tama?o del filtro gaussiano en px.
settings.filterSD   = 30;  % SD del filtro gaussiano en px.
settings.max_nhist  = 50;  % controla el colormap. ajustarlo para que no sature el heatmap.
settings.alpha      = 0.8; % entre 0 y 1. cuanto mayor, mas se ven las fijaciones.
settings.grayscale  = 1;
im_file             = '../data_analysis/utils/white.jpg';
im                  = imread(im_file);  
[sy,sx]             = size(im);

x = [smpinfo.x];
y = [smpinfo.y];
m = fun_heatmap(im_file, x, y, settings);
m = m/sum(m(:));

hstep = 32;
tr = 1;
    mar   = (smpinfo(tr).screen_size - smpinfo(tr).image_size)/2;
    XLIMI = [-mar(2) smpinfo(tr).image_size(2)+mar(2)]; %[-(hstep+0.5) (sx+hstep+0.5)];
    YLIMI = [-mar(1) smpinfo(tr).image_size(1)+mar(1)]; %[-(hstep+0.5) (sy+hstep+0.5)];
figure(1); clf
    set(gcf,'Color','w')
    subplot(3,3,[2 3 5 6])
        hold on
            imagesc(m)
            plot(x,y,'k.')
            plot([0 0],YLIMI,'k-')
            plot([sx sx],YLIMI,'k-')
            plot(XLIMI,[0 0],'k-')
            plot(XLIMI,[sy sy],'k-')
        hold off
        box on
        set(gca,'XLim',XLIMI,'XAxisLocation','Top')
        set(gca,'YLim',YLIMI,'YAxisLocation','Right')

    subplot(3,3,[1 4])
        [yh,xh] = hist(y,-hstep:hstep:(sy+hstep));
        hold on
            plot(yh,xh,'k-','LineWidth',3)
            plot(xlim,[0 0],'k-')
            plot(xlim,[sy sy],'k-')
        hold off
        box on
        set(gca,'XDir','reverse');
        set(gca,'XAxisLocation','Top')
        set(gca,'YLim',YLIMI,'YAxisLocation','Right','YTickLabel',[])
        
    subplot(3,3,[8 9])
        [yh,xh] = hist(x,-hstep:hstep:(sx+hstep));
        hold on
            plot(xh,yh,'k-','LineWidth',3)
            plot([0 0],ylim,'k-')
            plot([sx sx],ylim,'k-')
        hold off
        box on
        set(gca,'YDir','reverse');
        set(gca,'XLim',XLIMI,'XAxisLocation','Top','XTickLabel',[])
        set(gca,'YAxisLocation','Right')
end

%% Check: Overall fixation durations
if 1
d=[smpinfo.dur];

figure(2); clf
    set(gcf,'Color','w')
    subplot(2,2,1)
        [yh,xh] = hist(d,0:20:4000);
        hold on
            plot(xh,yh,'k-','LineWidth',3)
            plot([500 500],ylim,'k-')
        hold off
        box on
        xlabel('Fixation duration (ms)')
    subplot(2,2,2)
        [yh,xh] = hist(d,0:20:520);
        hold on
            plot(xh,yh,'k-','LineWidth',3)
        hold off
        box on
        xlim([0 500])
        xlabel('Fixation duration (ms)')
    subplot(2,2,3)
        [yh,xh] = hist(d,0:20:4000);
        hold on
            plot(log10(xh),yh,'k-','LineWidth',3)
            plot([log10(500) log10(500)],ylim,'k-')
        hold off
        box on
        xlabel('log10 Fixation duration (ms)')
    subplot(2,2,4)
        [yh,xh] = hist(d,0:20:520);
        hold on
            plot(log10(xh),yh,'k-','LineWidth',3)
        hold off
        box on
        xlim([log10(20) log10(500)])
        xlabel('log10 Fixation duration (ms)')
end

%% Check: Number of saccades vs number of saccades allowed
if 1
nfixs   = arrayfun(@(x) length(x.x),smpinfo);
nsaccs  = nfixs-1;
nsaccs_allowed  = [smpinfo.nsaccades_allowed];
N       = unique(nsaccs_allowed);
tf      = [smpinfo.target_found]==1;
rx      = 0.25*(2*rand(1,length(nsaccs_allowed))-1);
% ry      = zeros(1,length(nsaccs_allowed));
ry      = 0.25*(2*rand(1,length(nsaccs_allowed))-1);

figure(3); clf
    set(gcf,'Color','w')
    subplot(4,2,[1 3 5 7])
        hold on
            h(1)=plot(nsaccs_allowed(tf)+rx(tf),...
                        nsaccs(tf)+ry(tf),'r.','MarkerSize',10);
            h(2)=plot(nsaccs_allowed(~tf)+rx(~tf),...
                        nsaccs(~tf)+ry(~tf),'b.','MarkerSize',10);
            for n = 1:length(N)
                plot([0 13],[N(n) N(n)],'k-')
                plot([N(n) N(n)],[-0.25 14],'k-')
            end
        hold off
        legend(h,{'target found','target not found'},'location','best')
        xlabel('Number of saccades allowed')
        ylabel('Number of saccades')
        grid on
        box on
        ylim([-0.25 14])
        xlim([1 13])
        
    for n = 1:length(N)
        subplot(4,2,2*n)
            hold on
                [yh,xh] = hist(nsaccs(tf & nsaccs_allowed==N(n)),0:15);
                h(1)=plot(xh,yh,'r-','LineWidth',3);
                [yh,xh] = hist(nsaccs(~tf & nsaccs_allowed==N(n)),0:15);
                h(2)=plot(xh,yh,'b-','LineWidth',3);
                plot([N(n) N(n)],ylim,'k-')
            hold off
            box on
            set(gca,'XLim',[0 15],'XTick',N(n),'XTickLabels',{N(n)})
    end
        xlabel('Number of saccades')
end

%% Check: Response Times
if 1
rt  = [smpinfo.search_rt];
ort = [smpinfo.objetive_rt];
srt = [smpinfo.subjetive_rt];

figure(4); clf
    set(gcf,'Color','w')
    h=[];
    subplot(2,1,1)
    hold on
        [yh,xh] = hist(rt,0.25:.25:20); yh=yh/sum(yh);
            h(1)=plot(xh,yh,'k-','LineWidth',3);
        [yh,xh] = hist(ort,0.5:.5:20); yh=yh/sum(yh);
            h(2)=plot(xh,yh,'r-','LineWidth',3);
        [yh,xh] = hist(srt,0.5:.5:20); yh=yh/sum(yh);
            h(3)=plot(xh,yh,'b-','LineWidth',3);
    hold off
    box on
    legend(h,{'Search','Objective Response','Subjective Response'})
    xlabel('Time (s)')
    set(gca,'XLim',[0 10])
end

%%
if 1
p = nan(1,length(N));
for n = 1:length(N)
    p(n) = sum(tf(nsaccs_allowed==N(n)))/sum(nsaccs_allowed==N(n));
end

figure(5); clf
    set(gcf,'Color','w')
    plot(N,p,'k.-')
    box on
    set(gca,'XTick',N,'XLim',[1 13],'YLim',[0 1],'XGrid','on')
    xlabel('Number of saccades allowed')
    ylabel('P(target found)')
end

%% Check distribution of first fixation direction
if 1
    ind = arrayfun(@(x) ~isempty(x.x), smpinfo);
    xfirst  = arrayfun(@(x) x.x(1), smpinfo(ind));
    yfirst  = arrayfun(@(x) x.y(1), smpinfo(ind));
    xinit   = arrayfun(@(x) x.initial_position(1), smpinfo(ind));
    yinit   = arrayfun(@(x) x.initial_position(2), smpinfo(ind));

    dx = xfirst - xinit;
    dy = yfirst - yinit;
    mdx = median(dx);
    mdy = median(dy);

    th = 0:pi/50:2*pi;
    figure(10); clf
        set(gcf,'Color','w')
        subplot(1,2,1)
            hold on
                title(round(sqrt(mdx*mdx + mdy*mdy)))
                plot([-500 500],[0 0],'k--')
                plot([0 0],[-500 500],'k--')    
                for r = [50 100 200];
                    xunit = r * cos(th);
                    yunit = r * sin(th);
                    plot(xunit, yunit, 'k--');
                end
                for tr=1:length(dx)
                    plot([0 dx(tr)],[0 dy(tr)],'.-','Color',[.5 .5 .5])
                end
                plot([0 mdx],[0 mdy],'k.-','LineWidth',3,'MarkerSize',20)
            hold off
            box on
            axis square
            set(gca,'XLim',[-500 500])
            set(gca,'YLim',[-500 500])    
            
        subplot(1,2,2)
            hold on
                title(round(sqrt(mdx*mdx + mdy*mdy)))
%                 plot([-500 500],[0 0],'k--')
%                 plot([0 0],[-500 500],'k--')    
%                 for r = [50 100 200];
%                     xunit = r * cos(th);
%                     yunit = r * sin(th);
%                     plot(xunit, yunit, 'k--');
%                 end
                for tr=1:length(dx)
                    plot([xinit(tr) xfirst(tr)],[yinit(tr) yfirst(tr)],'.-','Color',[.5 .5 .5])
                    plot(xfirst(tr),yfirst(tr),'r.')
                    plot(xinit(tr),yinit(tr),'b.')
                end
%                 plot([0 mdx],[0 mdy],'k.-','LineWidth',3,'MarkerSize',20)
            hold off
            box on
            axis square
            hstep = 32;
            XLIMI = [-mar(2) smpinfo(tr).image_size(2)+mar(2)]; %[-(hstep+0.5) (sx+hstep+0.5)];
            YLIMI = [-mar(1) smpinfo(tr).image_size(1)+mar(1)]; %[-(hstep+0.5) (sy+hstep+0.5)];
            set(gca,'XLim',XLIMI,'XAxisLocation','Top')
            set(gca,'YLim',YLIMI,'YAxisLocation','Left','YDir','reverse')
end

%% Check: Scanpath: Target Found
if 1
trlist = find(tf == 1 & nsaccs_allowed==12 & nsaccs>8  & nsaccs<12);


addpath('../utils/')

hstep = 32;
XLIMI = [-mar(2) smpinfo(tr).image_size(2)+mar(2)]; %[-(hstep+0.5) (sx+hstep+0.5)];
YLIMI = [-mar(1) smpinfo(tr).image_size(1)+mar(1)]; %[-(hstep+0.5) (sy+hstep+0.5)];
tarcolor = 'r';
inicolor = 'b';

for tr = trlist
    figure(100+tr); clf;
        set(gcf,'Color','w')
        subplot(3,3,[1:6])
            hold on
                im  = imread(fullfile(dir_images,smpinfo(tr).image_name));   
                [sy,sx] = size(im);

                imagesc(repmat(im,[1 1 3]));
                plot([0 0],YLIMI,'k-')
                plot([sx sx],YLIMI,'k-')
                plot(XLIMI,[0 0],'k-')
                plot(XLIMI,[sy sy],'k-')

                r = smpinfo(tr).target_rect;
                xr= [r(1) r(1) r(3) r(3) r(1)]; 
                yr= [r(2) r(4) r(4) r(2) r(2)]; 
                patch(xr,yr,tarcolor,'FaceAlpha',.5,'EdgeColor',tarcolor)

                pxy = smpinfo(tr).initial_position;  
                plot(pxy(1),pxy(2),inicolor,'Marker','o','MarkerSize',10,'LineWidth',3)
                % OJO. Parece haber puntos de inicio fuera de la pantalla,
                % ver:
                % pxy = smpinfo(67).initial_position
                % pxy = [677 811]
                % pxy = smpinfo(84).initial_position
                % pxy = [814 534]
                
                spx = smpinfo(tr).x;% - dx(tr);
                spy = smpinfo(tr).y;% - dy(tr);
                spd = smpinfo(tr).dur;
                col = rainbow_colors(length(spx));
                plot(spx,spy,'w-')
                scatter(spx',spy',100*spd'/max(spd),col,'filled');

                sspx = smpinfo(tr).old.x_sacc;
                sspy = smpinfo(tr).old.y_sacc;
                scol = rainbow_colors(length(sspx));
                scatter(sspx',sspy',10, scol);
                
            hold off
            box on
            set(gca,'XLim',XLIMI,'XAxisLocation','Top')
            set(gca,'YLim',YLIMI,'YAxisLocation','Left','YDir','reverse')

        subplot(3,3,9)
            hold on
                tar = imread(fullfile(dir_templates,smpinfo(tr).target_name));   
                [sty,stx] = size(tar);

                imagesc(repmat(tar,[1 1 3]));
                plot([0 0],[-8 80],'k-')
                plot([stx stx],[-8 80],'k-')
                plot([-8 80],[0 0],'k-')
                plot([-8 80],[sty sty],'k-')
                plot([0 0],[0 sty],tarcolor,'LineStyle','-','LineWidth',3)
                plot([stx stx],[0 sty],tarcolor,'LineStyle','-','LineWidth',3)
                plot([0 stx],[0 0],tarcolor,'LineStyle','-','LineWidth',3)
                plot([0 stx],[sty sty],tarcolor,'LineStyle','-','LineWidth',3)
            hold off
            box on
            set(gca,'XLim',[-8 80],'XAxisLocation','Top')
            set(gca,'YLim',[-8 80],'YAxisLocation','Right','YDir','reverse')


end
end

%% Check: Scanpath: Target Not Found
if 0
trlist = find(tf == 0 & nsaccs_allowed==12);


addpath('../utils/')

hstep = 32;
XLIMI = [-mar(2) smpinfo(tr).image_size(2)+mar(2)]; %[-(hstep+0.5) (sx+hstep+0.5)];
YLIMI = [-mar(1) smpinfo(tr).image_size(1)+mar(1)]; %[-(hstep+0.5) (sy+hstep+0.5)];
tarcolor = 'r';
inicolor = 'b';

for tr = trlist
    figure(100+tr); clf;
        set(gcf,'Color','w')
        subplot(3,3,[1:6])
            hold on
                im  = imread(fullfile(dir_images,smpinfo(tr).image_name));   
                [sy,sx] = size(im);

                imagesc(repmat(im,[1 1 3]));
                plot([0 0],YLIMI,'k-')
                plot([sx sx],YLIMI,'k-')
                plot(XLIMI,[0 0],'k-')
                plot(XLIMI,[sy sy],'k-')

                r = smpinfo(tr).target_rect;
                xr= [r(1) r(1) r(3) r(3) r(1)]; 
                yr= [r(2) r(4) r(4) r(2) r(2)]; 
                patch(xr,yr,tarcolor,'FaceAlpha',.5,'EdgeColor',tarcolor)

                pxy = smpinfo(tr).initial_position;  
                plot(pxy(1),pxy(2),inicolor,'Marker','o','MarkerSize',10,'LineWidth',3)
                % OJO. Parece haber puntos de inicio fuera de la pantalla,
                % ver:
                % pxy = smpinfo(67).initial_position
                % pxy = [677 811]
                % pxy = smpinfo(84).initial_position
                % pxy = [814 534]
                
                spx = smpinfo(tr).x;
                spy = smpinfo(tr).y;
                spd = smpinfo(tr).dur;
                col = rainbow_colors(length(spx));
                plot(spx,spy,'w-')

                scatter(spx',spy',100*spd'/max(spd),col,'filled');

            hold off
            box on
            set(gca,'XLim',XLIMI,'XAxisLocation','Top')
            set(gca,'YLim',YLIMI,'YAxisLocation','Left','YDir','reverse')

        subplot(3,3,9)
            hold on
                tar = imread(fullfile(dir_templates,smpinfo(tr).target_name));   
                [sty,stx] = size(tar);

                imagesc(repmat(tar,[1 1 3]));
                plot([0 0],[-8 80],'k-')
                plot([stx stx],[-8 80],'k-')
                plot([-8 80],[0 0],'k-')
                plot([-8 80],[sty sty],'k-')
                plot([0 0],[0 sty],tarcolor,'LineStyle','-','LineWidth',3)
                plot([stx stx],[0 sty],tarcolor,'LineStyle','-','LineWidth',3)
                plot([0 stx],[0 0],tarcolor,'LineStyle','-','LineWidth',3)
                plot([0 stx],[sty sty],tarcolor,'LineStyle','-','LineWidth',3)
            hold off
            box on
            set(gca,'XLim',[-8 80],'XAxisLocation','Top')
            set(gca,'YLim',[-8 80],'YAxisLocation','Right','YDir','reverse')


end
end

%%
if 0
% Si encontro el target, quiero ver que: 1. Haya una fijacion al target y
% 2. Si fue la ultima
clc
N_fix2target = nan(1,length(smpinfo));
for tr = 1:length(smpinfo)
    if (smpinfo(tr).target_found==1)
        fix2target = ( smpinfo(tr).old.x >= smpinfo(tr).target_rect(1)-M & ...
                smpinfo(tr).old.x <= smpinfo(tr).target_rect(3)+M) & ...
            ( smpinfo(tr).old.y >= smpinfo(tr).target_rect(2)-M & ...
                smpinfo(tr).old.y <= smpinfo(tr).target_rect(4)+M );
        
        if sum(fix2target)>0
            N_fix2target(tr) = length(fix2target)-find(fix2target,1,'first');
        else
            N_fix2target(tr) = 99;
        end            
    end
end

disp(sum(~isnan(N_fix2target)))
disp(sum(N_fix2target==99))
hh = hist(N_fix2target(N_fix2target~=99 & ~isnan(N_fix2target)),0:10);
end

%%
% Volver para atras, encontrar la posicion del ojo en la fijacion previa y
% promediar para comparar con la esperada
load([dirtodo subjnames{su} '_todo.mat']);

indfix = find(cellfun(@(x) ~isempty(x),strfind(todo.msg,'fixcross')));
indsrt = find(cellfun(@(x) ~isempty(x),strfind(todo.msg,'startsearch')));

disp([length(indfix) length(indsrt)])

fixtimes = [todo.msgtime(indfix) todo.msgtime(indsrt)];

N_full_fix      = nan(length(indfix),1);
last_full_fix_x = nan(length(indfix),1);
last_full_fix_y = nan(length(indfix),1);
last_parc_fix_x = nan(length(indfix),1);
last_parc_fix_y = nan(length(indfix),1);
for tr = 1:length(indfix)
    ind_full_fix    = ( todo.refix(:,1) >= fixtimes(tr,1) & todo.refix(:,2) <= fixtimes(tr,2) );
    Nf(tr)          = sum( ind_full_fix );
   
    mar   = (smpinfo(tr).screen_size - smpinfo(tr).image_size)/2;
    
    if ( Nf(tr)>0 )
        indlast     = find( ind_full_fix , 1, 'last');
        last_full_fix_x(tr) = todo.refix(indlast,4) - mar(1);
        last_full_fix_y(tr) = todo.refix(indlast,5) - mar(2);
    end
    
    
    indlast     = find( todo.refix(:,1) >= fixtimes(tr,1) & ...
                        todo.refix(:,1) < fixtimes(tr,2) & ...
                        todo.refix(:,2) > fixtimes(tr,2), 1, 'first');
    if ( ~isempty(indlast) )
        last_parc_fix_x(tr) = todo.refix(indlast,4) - mar(1);
        last_parc_fix_y(tr) = todo.refix(indlast,5) - mar(2);
    end
end

xinit   = arrayfun(@(x) x.initial_position(1), smpinfo(ind));
yinit   = arrayfun(@(x) x.initial_position(2), smpinfo(ind));

for ordenados = 0:1
    figure; clf;
        dx1 = last_full_fix_x-xinit';
        dy1 = last_full_fix_y-yinit';
        dd1 = sqrt(dx1.^2 + dy1.^2);
        dx2 = last_parc_fix_x-xinit';
        dy2 = last_parc_fix_y-yinit';
        dd2 = sqrt(dx2.^2 + dy2.^2);

        subplot(2,2,1)
            hold on
                plot([0 135],[0 0], 'k-')
                if ordenados
                    plot(sort(dx1), 'r-','LineWidth',2)
                    plot(sort(dy1), 'b-','LineWidth',2)
                    plot(sort(dx2), 'r--','LineWidth',2)
                    plot(sort(dy2), 'b--','LineWidth',2)
                else
                    plot(dx1, 'r-','LineWidth',2)
                    plot(dy1, 'b-','LineWidth',2)
                    plot(dx2, 'r--','LineWidth',2)
                    plot(dy2, 'b--','LineWidth',2)
                end
            hold off
            box on
            xlim([0 135])
        subplot(2,2,3)
            hold on
                if ordenados
                    plot(sort(dd1), 'k-','LineWidth',2)
                    plot(sort(dd2), 'k--','LineWidth',2)
                else
                    plot(dd1, 'k-','LineWidth',2)
                    plot(dd2, 'k--','LineWidth',2)
                end
            hold off
            box on
            xlim([0 135])

        dx3 = min([dx1,dx2]')';
        dy3 = min([dy1,dy2]')';
        dd3 = min([dd1,dd2]')';
        subplot(2,2,2)
            hold on
                plot([0 135],[0 0], 'k-')
                if ordenados
                    plot(sort(dx3), 'r-','LineWidth',2)
                    plot(sort(dy3), 'b-','LineWidth',2)
                else
                    plot(dx3, 'r-','LineWidth',2)
                    plot(dy3, 'b-','LineWidth',2)
                end
            hold off
            box on
            xlim([0 135])
        subplot(2,2,4)
            hold on
                if ordenados
                    plot(sort(dd3), 'k-','LineWidth',2)
                else
                    plot(dd3, 'k-','LineWidth',2)
                end
            hold off
            box on
            xlim([0 135])
end
        
        