%% Inverse Source Problem of 2D heat PDE with different methods
% Fast matrix exponential-based quasi-boundary value methods for
% inverse space-dependent source problems,
% by Fermın Bazan, Luciano Bedin, Koung Hee Leeem, Jun Liu *, George Pelekanos.
clear 
methodlist={'QBVM','MQBVM','MatExp-a','MatExp-b'}; 
tl=tiledlayout(2,2,'TileSpacing','Compact'); 
for kk=1:4
method=methodlist{kk};
maxL=5; % increase for larger mesh, can be very slow for all-at-once QBVM
noiselist=10.^(-3); 
errtab=[]; cputab=[];
lsty ={':','-.','--','.-','x-','d-','^-','v-','>-','<-','p-','h-','.-'};
fprintf('=============Method[%s]==========\n',method);
y0fun=@(x,y) zeros(size(x)); %gives zero initial condition
%Uncomment each example to test
xa=0; xb=pi; T=1;    %square domain [0,pi]^2
%% 2D Example 4, smooth example
 exname='Ex4';  f=@(x,y)  (x-xa).*(xb-x).*sin(2*x).*(y-xa).*(xb-y).*cos(y);   

for k=1:length(noiselist)
    noiselev=noiselist(k); 
    nxlist=2.^(4:maxL); ntlist=T*nxlist;
    levmax=length(nxlist); 
    errlist=[];cpulist=[]; 
    for s=1:levmax        
        nx=nxlist(s);m=nx-1; nt=ntlist(s);
        dt=T/nt; h=(xb-xa)/nx;  
        xx=xa+h:h:xb-h; yy=xa+h:h:xb-h;
        [XX,YY]=ndgrid(xx,yy); 
        b_y=zeros(m,m,nt+1); %RHS  of system     
        Lap=(1/h^2)*gallery('poisson',m); %negative Laplacian          
        %construct system matrix based on the method
        Ix=speye(m^2); It=speye(nt+1);e=ones(nt+1,1);
        B=spdiags([-e e]/dt,[-1 0],nt+1,nt+1);   
        %Use CN time-marching scheme to compute final time condition
        fx=f(XX,YY); yh_T=y0fun(XX,YY);%can be any IC
        for jj=1:nt
            yh_T(:)=(Ix/dt+Lap/2)\(fx(:)+(Ix/dt-Lap/2)*yh_T(:));
        end         
        %add noise to the final time observation
        b_y(:,:,1)=yh_T.*(1+noiselev*(-1+2*rand(m,m))); %uniform noise [-1,1]
        delta=h*norm(b_y(:,:,1)-yh_T,'fro');   %noise level
        mm=m^2;alpha=0;beta=0; 
        switch(method)
            case 'MatExp-a'
                alpha=sqrt(delta);
            case 'MatExp-b'
                beta=delta; 
        end
        switch(method)
            case 'QBVM' %QBVM
                beta=sqrt(delta); 
                It0=It; It0(1,1)=0; B(1,1)=beta;B(1,end)=1;B(2:end,1)=-1;
                L=kron(B,Ix)+kron(It0,Lap); %can not be fit PinT way, due to It0 not an identity matrix
                tic; x=L\b_y(:); tsolving=toc; %sparse direct solver   
            case 'MQBVM'  %MQBVM with alpha=0
                beta=delta; alpha=0;
                B(1,1)=alpha;B(1,end)=1/beta;B(2:end,1)=-1; b_y(:,:,1)=b_y(:,:,1)/(beta);        
                L=kron(B,Ix)+kron(It,Lap);  tic;x=L\b_y(:); tsolving=toc; %sparse direct solver              
            case {'MatExp-a','MatExp-b'}
                %beta=sqrt(delta);
                g=b_y(:,:,1); phi=y0fun(XX,YY);
                tic;
                %expA=expm(-T*Lap); %expensive to construct explicitly
                %rhs=Lap*(g(:)-expA*phi(:));  
                %x=((Ix-expA)+beta*Lap)\rhs; %regularizaion term: beta*Lap
                %% itertive solver
                rhs=Lap*(g(:)-expmv(T,-Lap,phi(:),[],'half')); 
                Afun=@(z) z+alpha*Lap*z+beta*Lap*(Lap*z)-expmv(T,-Lap,z,[],'half'); 
                [x,flag,relres,iter]=pcg(Afun,rhs,1e-3,50); %PCG with max 50 iterations and tolerance=1e-3
                tsolving=toc;  
        end            
        cpulist=[cpulist;tsolving];
        %%Measure errors:     
        f_h=reshape(x(1:mm),m,m); 
        f_err=h*norm(f_h-fx,'fro');%L2 norm 
        errlist=[errlist;f_err];y_err0=f_err;
        %plot errors for different noise level with the finest mesh
        if(s==levmax)
            nexttile 
            mesh(XX,YY,f_h); hold on; axis tight
            xlabel('$x$'); ylabel('$y$'); zlabel('$f_h(x,y)$')
            title(sprintf('%s($\\delta$=%1.1E,error=%1.2E,CPU=%1.2f)',method,delta,f_err,tsolving)) 
            set(gca(),'FontSize',20); hold on
            drawnow
        end       
    end
    errtab=[errtab errlist];
    cputab=[cputab cpulist];
end
fprintf('-------------------------Errors-----------------------\n');
errtab=[nxlist' ntlist' errtab];
fprintf(['&($%3.0d^2$,%4.0d)\t' repmat(' &%1.2e \t', 1, length(noiselist)) '\\\\\n'], errtab'); %h vs delta

fprintf('-------------------------CPU-----------------------\n');
cputab=[nxlist' ntlist' cputab];
fprintf(['($%3.0d^2$,%4.0d)\t' repmat(' &%1.2f \t', 1, length(noiselist)) '\\\\\n'], cputab'); %h vs delta
end  
figname1=sprintf('../Regularization/%s_mesh_%d',exname,maxL);
set(gcf, 'Position', get(0, 'Screensize').*[1 1 1 1]);
%legend('-DynamicLegend','Location', 'SouthEast');
%print(figname1,'-depsc')
 