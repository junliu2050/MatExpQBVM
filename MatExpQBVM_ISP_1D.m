%% Inverse Source Problem of 1D heat PDE with different methods
% Fast matrix exponential-based quasi-boundary value methods for
% inverse space-dependent source problems,
% by Fermın Bazan, Luciano Bedin, Koung Hee Leeem, Jun Liu *, George Pelekanos.
clear 
methodlist={'QBVM','MQBVM','MatExp-a','MatExp-b'};
tl=tiledlayout(2,2,'TileSpacing','Compact');
for kk=1:4 %loop each method
    method=methodlist{kk};
    maxL=10; % increase for larger mesh size
    noiselist=10.^(-1:-1:-3);
    errtab=[]; cputab=[];
    lsty ={':','-.','--','.-','x-','d-','^-','v-','>-','<-','p-','h-','.-'};
    fprintf('=============Method[%s]==========\n',method);
    y0fun=@(x) zeros(size(x)); %gives zero initial condition
    xa=0; xb=pi;T=1;
    %Uncomment each example to test different examples
    %% Example 1
    exname='Ex1'; f=@(x) (x-xa).*(xb-x).*sin(4*x);  %space-dependent only source term
    %% Example 2
    exname='Ex2'; f=@(x) 2*x.*(x<=pi/2)+2*(pi-x).*(x>pi/2); %Triangle
    %% Example 3
    exname='Ex3'; f=@(x) 1*((x>=pi/3)&(x<=2*pi/3)); %discontinous
    for k=1:length(noiselist)
        noiselev=noiselist(k);
        nxlist=2.^(5:maxL); ntlist=T*nxlist;
        levmax=length(nxlist);
        errlist=[];cpulist=[];condVlist=[];
        for s=1:levmax
            nx=nxlist(s);m=nx-1; nt=ntlist(s);
            dt=T/nt; h=(xb-xa)/nx; xx=xa+h:h:xb-h;
            b_y=zeros(m,nt+1); %RHS
            Lap=(1/h^2)*gallery('tridiag',m,-1,2,-1); %negative Laplacian
            %construct system matrix based on the method
            Ix=speye(m); It=speye(nt+1); e=ones(nt+1,1);
            B=spdiags([-e e]/dt,[-1 0],nt+1,nt+1);
            %Use CN time-marching scheme to compute final time condition
            fx=f(xx'); yh_T=y0fun(xx');%can be any IC
            for jj=1:nt
                yh_T=(Ix/dt+Lap/2)\(fx+(Ix/dt-Lap/2)*yh_T);
            end
            %add random noise to the final time observation
            b_y(:,1)=yh_T.*(1+noiselev*(-1+2*rand(m,1))); %uniform noise [-1,1]
            delta=sqrt(h)*norm(b_y(:,1)-yh_T); %E=sqrt(h)*norm(fh);
            %%choice of regularization parameter
            alpha=0;beta=0;
            switch(method)
                case 'MatExp-a'
                    alpha=sqrt(delta);
                case 'MatExp-b'
                    beta=delta;
            end
            switch(method)
                case 'QBVM' %QBVM based on all-at-once scheme
                    alpha=sqrt(delta);
                    It0=It; It0(1,1)=0; B(1,1)=alpha;B(1,end)=1;B(2:end,1)=-1;
                    L=kron(B,Ix)+kron(It0,Lap); %construct all-at-once system
                    tic; x=L\b_y(:); tsolving=toc; %sparse direct solver, slow for large size
                case 'MQBVM'  %MQBVM with alpha=0
                    beta=delta;
                    B(1,1)=alpha;B(1,end)=1/beta;B(2:end,1)=-1; b_y(:,1)=b_y(:,1)/(beta);%rhs/beta
                    L=kron(B,Ix)+kron(It,Lap); %construct all-at-once system
                    tic;x=L\b_y(:); tsolving=toc; %sparse direct solver
                case {'MatExp-a','MatExp-b'} %based on the system (4.7)
                    g=b_y(:,1); %final measurement with noise
                    phi=y0fun(xx');%initial condition
                    tic;
                    expA=expm(-T*Lap); %expensive to construct explicitly, only for 1D case
                    rhs=Lap*(g-expA*phi);
                    x=((Ix-expA)+alpha*Lap+beta*Lap*Lap)\rhs; %regularizaion term: beta*Lap 
                    tsolving=toc;
            end
            cpulist=[cpulist;tsolving];
            %%Measure errors:
            f_h=x(1:m); f_err=sqrt(h)*norm(f_h-fx);%L2 norm
            errlist=[errlist;f_err]; y_err0=f_err;
            %plot errors for different noise level with the finest mesh
            if(s==levmax)
                nexttile(kk) 
                xgrid=[xa xx xb];
                if(k==1)
                    plot(xgrid,[0;fx;0],'k-','LineWidth',2,'DisplayName','Exact'); hold on;
                end
                if(alpha>0)
                    plot(xgrid,[0;f_h;0],lsty{k},'LineWidth',2,'DisplayName',sprintf('\\delta=%1.1E,\\alpha=%1.1E',delta,alpha)); hold on;
                else
                    plot(xgrid,[0;f_h;0],lsty{k},'LineWidth',2,'DisplayName',sprintf('\\delta=%1.1E,\\beta=%1.1E',delta,beta)); hold on
                end
                axis tight
                xlabel('$x$'); ylabel('$f(x)$');
                title(sprintf('%s (error=%1.2E,CPU=%1.2f s)',method,f_err,tsolving))
                legend(gca,'show','Location','NorthEast');
                set(gca(),'FontSize',20); hold on
            end
        end
        errtab=[errtab errlist];
        cputab=[cputab cpulist];
    end

    fprintf('-------------------------Errors-----------------------\n');
    errtab=[nxlist' ntlist' errtab];
    fprintf(['&(%3.0d,%4.0d)\t' repmat(' &%1.2e \t', 1, length(noiselist)) '\\\\\n'], errtab'); %h vs delta

    fprintf('-------------------------CPU-----------------------\n');
    cputab=[nxlist' ntlist' cputab];
    fprintf(['(%3.0d,%4.0d)\t' repmat(' &%1.2f \t', 1, length(noiselist)) '\\\\\n'], cputab'); %h vs delta
end
figname1=sprintf('../Regularization/%s_mesh_%d',exname,maxL);
set(gcf, 'Position', get(0, 'Screensize').*[1 1 1 1]);
legend('-DynamicLegend','Location', 'NorthEast');
%print(figname1,'-depsc')


