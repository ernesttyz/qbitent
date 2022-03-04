function []=qbitent_demo()
%This is a demonstration of the computation of a lower bound on H(A_X|E).
%The main function in this work is qbitent(); the following lines just set up an example of input values that can be supplied to it.

refCHSH=2; %The entropy bound will be computed for this CHSH value, but the choice is somewhat arbitrary since in any case we will just use it as a "reference point" for the linear bound.
corrs = refCHSH*[1 1 1 -1]/4; %Set of correlators compatible with the CHSH value specified above
lambdas = .139*[1 1 1 -1]; %Optimal lambda according to heuristics
pNPPsame = .3; pNPP = [pNPPsame pNPPsame]; %Uses the same value of noisy preprocessing for both A0 and A1
pA0 = .5; %Same probability for both bases
rngseed=0;
optmode=1; prec=0.0003; %This choice with gridnum=48 and an initial grid of 150 points ensures the refinement depth is at most 1 
gridnum=48; parnum=48; %Suitable choice for Euler cluster
breakval=shannon([pNPPsame 1-pNPPsame])+contbndA(lambdas,prec) %Allows early FW termination; see below. Warning: this choice is only consistent with refCHSH=2. It is a conservative choice, which is probably better - a suboptimal bound at one data point will later need to be refined, which costs more time later. 

%Main computation of lower bound on H(A_X|E)
entrate = qbitent(corrs,lambdas,pNPP,pA0,rngseed,optmode,prec,gridnum,parnum,breakval)

end

function entrate = qbitent(corrs,lambdas,pNPP,pA0,rngseed,optmode,prec,gridnum,parnum,breakval)
%This code may encounter numerical issues with probabilities exactly equal to 0 (e.g. when computing entropy), so avoid extremal inputs if possible.
%Computes a lower bound on H(A_X|E).
%corrs is a 1x4 array specifying the correlators <A0B0>,<A0B1>,<A1B0>,<A1B1> in that order (by the arguments in our paper, we take the marginals to be uniform).
%lambdas is a 1x4 array specifying the Lagrange multipliers (in the same order as the correlators).
%pNPP is a 1x2 array, with each entry in [0,0.5], specifying the amount of noisy preprocessing to apply to measurements A0 and A1 respectively (we use the convention that it denotes the probability of flipping the outcome).
%pA0 is a scalar in [0,0.5] specifying the probability with which basis A0 is selected in key-generation rounds.
%rngseed is either [] or a valid RNG seed. If [], it does nothing, otherwise it seeds the Matlab RNG with that value.
%optmode is a value in {1,2,3,4} that switches between certified and heuristic methods. If 1 or 2, the iterative refinement methods are used for the angles and either Frank-Wolfe / heuristic methods (respectively) for the state (the latter is convex, so it usually converges closely to the true optimum). If 3 or 4, heuristic minimization is used for all parameters, and in the latter case with the further constraint that only two Bell eigenvalues are nonzero.
%prec is either a positive integer or a 1x2 array. A positive-integer value should be used iff optmode=3 or 4, in which case it specifies the number of attempts at heuristic minimization (the smallest value is returned). Otherwise, the array specifies the desired upper bounds on the smallest grid spacings for Alice and Bob's measurement angles, i.e. the algorithm terminates when the grid spacings in both angles have been reduced to values smaller than prec.
%parnum is either [] or an integer. If [], no parallel computing is used, otherwise it specifies the number of threads to set up in parpool.
%gridnum is an integer that specifies the number of parallel threads invoked in parfor, if used.
%breakval is a float that allows early termination of FW algorithm; the algorithm terminates if it finds a certified bound greater than breakval.

% Adjust as needed to include YALMIP and MOSEK
addpath(genpath('YALMIP-master'))
addpath(genpath('mosek'))

if isempty(rngseed)
    %Do nothing
else
    rng(rngseed); 
end


if optmode == 1 || optmode == 2
    difftols=[5e-4 5e-4]; maxits=[200 200];
    maxtime=119*3600;
    [optval fulldata errlist] = dualfunc(corrs,lambdas,pNPP,pA0,maxtime,optmode,prec,difftols,breakval,maxits,gridnum,parnum);
    errlist
elseif optmode == 3 || optmode==4
    %In this block, the parametrization used is to keep the state Bell-diagonal and allow all measurements to be in arbitrary directions in the XZ plane.
    %This parametrization apparently helps the heuristic algorithms to converge. However, it means I need to define the objective using a somewhat different set of functions, instead of directly using those written for the above block.
    perturb=1e-14; %Small perturbation to avoid taking relative entropy of singular matrix
    %The following approach is a bit odd, but allows the functions below to be phrased in terms of just 1 Lagrange multiplier
    %In any case, this approach very slightly saves time by precomputing the coefficient matrix instead of recomputing it with each function call
    coeffslist = {coeffscorr2(1,1),coeffscorr2(1,2),coeffscorr2(2,1),coeffscorr2(2,2)}; coeffs = zeros(2,2,2,2); 
    lambgam = 0; % The lambda*gamma term
    for j=1:4
        coeffs = coeffs + lambdas(j)*coeffslist{j};
        lambgam = lambgam + lambdas(j)*corrs(j);
    end
    % Following up on the above, the next part sets lambda=1 and gamma=lambgam in the function arguments (since lambda=1 implies gamma=lambgam/lambda=lambgam).
    if optmode==3
        fminobj=@(tup)real(dualobj2(tup,coeffs,1,lambgam,pNPP,pA0,perturb)); 
        vecsize=7;numtrials=prec; 
        Aineq = [1 1 1 0 0 0 0]; bineq = [1]; 
        Aeq = []; beq = [];
        lb = [0 0 0 0 0 0 0];
        ub = [1 1 1 2*pi 2*pi 2*pi 2*pi];
    else
        fminobj=@(tup)real(dualobj2([tup(1) 0 1-tup(1) tup(2:5)],coeffs,1,lambgam,pNPP,pA0,perturb)); 
        vecsize=5;numtrials=prec; 
        Aineq = []; bineq = []; 
        Aeq = []; beq = [];
        lb = [0 0 0 0 0];
        ub = [1 2*pi 2*pi 2*pi 2*pi];
    end
    sols = zeros(numtrials,vecsize); vals = 100*ones(numtrials,1);
    for trial=1:numtrials
        tup0 = rand(1,vecsize);
        options = optimset('Display','off');
        [sols(trial,:),vals(trial)] = fmincon(fminobj,tup0,Aineq,bineq,Aeq,beq,lb,ub,[],options);
    end
    [optval, optvalpos] = min(vals);
    optsol = sols(optvalpos,:);
else
    error('Invalid optmode')
end

entrate = optval;

end

%___________________________________________________________________

function [optval fulldata errlist] = dualfunc(corrs,lambdas,pNPP,pA0,maxtime,optmode,prec,difftols,breakval,maxits,gridnum,parnum)
% Bounds the minimum by iteratively refining a grid of values for the measurement angles.
% maxtime specifies the maximum time for which this function will run (upon timing out, it will return the data it generated up to that point, which may not be sufficiently refined to yield a good bound but will be a secure lower bound).
% optmode must be a value in {1,2}, with functionalities as defined above.
% prec,gridnum,parnum are as defined above.
% difftol (float), maxits (positive integer), breakval (float) are only used when optmode=1; they specify termination conditions for the FW algorithm (for details, see definition of minFW() below). 
% Returns optval (smallest (i.e. worst-case) certified bound in all the evaluated points), fulldata (all values of thA and (r_X,r_Z) that were evaluated), and errlist (list of any errors that occured in replacevert(), the vertex-replacing function).
% Note that for optmode=1 the states are the states that yield the certified bounds, while for optmode=2 they are the states that yield the feasible values.

if not(isempty(parnum))
    poolobj = gcp('nocreate'); delete(poolobj); %Clear any existing pool
    myCluster = parcluster('local'); 
    myCluster.NumWorkers = parnum; saveProfile(myCluster); %Increase NumWorkers parameter
    parpool(myCluster,parnum)
end

% Note that in each iteration, the worst point is deleted completely rather than simply having its continuity correction recomputed. This costs one extra computation, but may help to "unstick" points where the FW algorithm did not give a good lower bound.
% As mentioned above, each row of griddata essentially specifies [thA deltA bound-contbndA]
inisize=150; inidelt=(pi/2)/inisize; griddata = [linspace(inidelt,pi-inidelt,inisize)' ones(inisize,1)*[inidelt -10]]; %Seeding the algorithm with a fixed set of gridpoints
format long
griddata
format short
fulldata = {}; errlist = [];
worstpos=1; worstdata=griddata(worstpos,:); mindelt = worstdata(2);
perturb=1e-14; %Small perturbation to avoid taking relative entropy of singular matrix
initime = clock;
loopcount=0;
while mindelt > prec
    
    griddata(worstpos,:)=[]; %Deletes worst value
    olddelt=worstdata(2); newdelt=olddelt/gridnum; thAs = worstdata(1)+linspace(-olddelt+newdelt,olddelt-newdelt,gridnum);
    newdata = [thAs' newdelt*ones(gridnum,1) zeros(gridnum,1)];
    newresults=cell(1,gridnum);
    datetime('now')
    parfor j=1:gridnum
        if optmode==1
            warning('off','MATLAB:nearlySingularMatrix')
        end
        thA=thAs(j);
        
        % First perform a heuristic computation to get an estimate of the required polygonal approximation
        % Since the optimization is convex, in theory a single trial should suffice for heuristic optimizations, but sometimes it does not perform that well
        notused=-1; %Placeholder value for variables which should not be relevant
        estoptmode=2; estdifftols=[1e-3 notused]; estmaxits=[200 notused]; inidata=[0 1; 1 1; 1 -1; 0 -1]; heunumtrials=3; 
        [notused1 heupoldata notused2] = dualfuncthA(thA,corrs,lambdas,pNPP,pA0,inidata,estoptmode,perturb,estdifftols,[],estmaxits,heunumtrials)
        
        inidata=heupoldata;
        heunumtrials=1;

        [optboundA poldata newerrlist]=dualfuncthA(thA,corrs,lambdas,pNPP,pA0,inidata,optmode,perturb,difftols,breakval,maxits,heunumtrials)
        
        newresults{j} = {thA poldata newerrlist};
        newdata(j,3) = optboundA-contbndA(lambdas,newdelt);
        
        warning('on','MATLAB:nearlySingularMatrix')
    end
    
    for j=1:gridnum
        fulldata{end+1} = {newresults{j}{1} newresults{j}{2}};
        errlist = [errlist; newresults{j}{3}];
    end
    
    newdata=real(newdata);
    griddata = [griddata; newdata];
    [worstval worstpos] = min(griddata(:,end)); %min() only returns the position of one minimum (the first), which suits our purposes
    worstdata=griddata(worstpos,:);
    mindelt=worstdata(2); 
    
    % Display outputs after each parfor loop
    newdata
    [worstdata worstdata(end)+contbndA(lambdas,worstdata(2))]
    
    save('currentstate.mat')
    
    if etime(clock,initime) > maxtime
        break
    end
    
end
dlmwrite(['griddata' num2str(floor(pNPP(1)*100)) '.csv'],griddata,'precision','%.15f');
save('currentstate.mat');
optval = worstval;
end

function contbndA = contbndA(lambdas,deltA)
% Warning: must have deltA < pi to be valid
contbndA = 2.012/2*deltA + (abs(lambdas(3))+abs(lambdas(4)))*sqrt(2-2*cos(deltA));
end

function [optval poldata errlist] = dualfuncthA(thA,corrs,lambdas,pNPP,pA0,inidata,optmode,perturb,difftols,breakval,maxits,heunumtrials)
difftolFW=difftols(2); maxitFW=maxits(2);

fminobj=@(tup)real(dualobj(tup(1:5),thA,[sin(tup(6)) cos(tup(6))],corrs,lambdas,pNPP,pA0,perturb));
reftrials=5;
vecsize=6; sols = zeros(reftrials,vecsize); vals = 100*ones(reftrials,1);
for trial=1:reftrials
    tup0 = rand(1,6)/4; 
    Aineq = [1 1 1 0 0 0]; bineq = [1]; 
    Aeq = []; beq = [];
    lb = [0 0 0 -2 -2 0];
    ub = [1 1 1 2 2 pi];
    nonlcon = @(tup)negmineig(tup(1:5));
    options = optimset('Display','off');
    [sols(trial,:),vals(trial)] = fmincon(fminobj,tup0,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options);
end
reffeasval=min(vals);

% Heuristic computations suggest that (for a fixed thA) the minimum is often attainable by thB=pi/2 (for the range of lambdas we consider), so we might as well exploit that
newreffeasval = heudualfuncth(thA,[1 0],corrs,lambdas,pNPP,pA0,perturb,10);
if newreffeasval<reffeasval
    reffeasval=newreffeasval;
end

maxtries=10;

poldata = [inidata(:,1:2) zeros(size(inidata,1),7)];
if optmode==1
    for j=1:size(poldata,1)
        vecB=poldata(j,1:2); 
        if size(inidata,2)==2 %If inidata does not come with values+states corresponding to the vertices, we will generate them
            seeddat=[];
        elseif size(inidata,2)==9 %If inidata comes with values+states corresponding to the vertices, we will use them as seeds for FW algorithm
            seeddat=inidata(j,3:8);
        else
            error('Invalid inidata size!')
        end
        [bestcertval bestfeasval bestcertsol bestfeassol certval feasval] = minFW(thA,vecB,corrs,lambdas,pNPP,pA0,perturb,difftolFW,seeddat,breakval,maxitFW);
        poldata(j,3:9) = real([bestfeasval bestcertsol bestcertval]); %Occasionally small imaginary components show up
    end
elseif optmode==2 %For optmode=2 we will not use values+states in inidata, even if present
    for j=1:size(poldata,1)
        vecB=poldata(j,1:2); 
        thresh=heudualfuncth(thA,vecB/norm(vecB),corrs,lambdas,pNPP,pA0,perturb,3); 
        if thresh<reffeasval %Can use threshold value to update reference feasible value
            reffeasval=thresh;
        end
        [optfeasval optstate] = threshdualfuncth(thresh,maxtries,thA,vecB,corrs,lambdas,pNPP,pA0,perturb,heunumtrials); %Each computation of this line takes between 0.2 and 1 seconds
        poldata(j,3:9) = [optfeasval optstate optfeasval];
    end
else
    error('Invalid optmode!')
end

[worstval worstpos] = min(poldata(:,end)); %min() only returns the position of one minimum (the first), which suits our purposes
worstdata=poldata(worstpos,:);
errlist=[];
for iter=1:maxits(1)
    
    [newpts errlistnew] = replacevert(worstdata(1:2));
    errlist = [errlist; errlistnew];
    if newpts(1,:)==newpts(2,:) %Special case: If the replaced vertex is already on the unit circle, replacevert() would simply return the input. In that case we shall just recompute for that point (in the hope that the minimization will converge better on a second run).
        newdata=[newpts(1,:) zeros(1,7)];
    else
        newdata=[newpts zeros(2,7)];
    end
    
    if optmode==1
        if size(newdata,1)==1 %Following up on special case from above, if we recompute the point then we will "continue" the FW algorithm from the previous result.
            vecB=newdata(1:2); seeddat = worstdata(3:8);
            newdifftolFW=difftolFW/2; %Will aim to have the FW algorithm converge to a tighter tolerance on the continued computation.
            [bestcertval bestfeasval bestcertsol bestfeassol certval feasval] = minFW(thA,vecB,corrs,lambdas,pNPP,pA0,perturb,newdifftolFW,seeddat,breakval,maxitFW);
            newdata(3:9) = [bestfeasval bestcertsol bestcertval];
        else
            for pt=1:size(newdata,1)
                vecB=newdata(pt,1:2); seeddat = [];
                [bestcertval bestfeasval bestcertsol bestfeassol certval feasval] = minFW(thA,vecB,corrs,lambdas,pNPP,pA0,perturb,difftolFW,seeddat,breakval,maxitFW);
                newdata(pt,3:9) = [bestfeasval bestcertsol bestcertval];
            end
        end
    elseif optmode==2
        for pt=1:size(newdata,1)
            vecB=newdata(pt,1:2); 
            thresh=heudualfuncth(thA,vecB/norm(vecB),corrs,lambdas,pNPP,pA0,perturb,3); 
            if thresh<reffeasval %Can use threshold value to update reference feasible value
                reffeasval=thresh;
            end
            [optfeasval optstate] = threshdualfuncth(thresh,maxtries,thA,vecB,corrs,lambdas,pNPP,pA0,perturb,heunumtrials);
            newdata(pt,3:9) = [optfeasval optstate optfeasval];
        end
    end
    newdata = clean(newdata,1e-35); %Occasionally small imaginary components show up
    poldata = [poldata(1:worstpos-1,:); newdata; poldata(worstpos+1:end,:)];
    [worstval worstpos] = min(poldata(:,end)); %min() only returns the position of one minimum (the first), which suits our purposes
    worstdata=poldata(worstpos,:);

    vecB=worstdata(1:2);
    if optmode==1 %Only necessary for optmode 1, because in optmode 2 it's updated above
        newreffeasval=heudualfuncth(thA,vecB/norm(vecB),corrs,lambdas,pNPP,pA0,perturb,heunumtrials); %Heuristically computing a potential new feasible value
        if newreffeasval<reffeasval
            reffeasval=newreffeasval;
        end
    end
    
    if reffeasval-worstval < difftols(1) && norm(vecB)-1<1e-2 %Strictly speaking the latter is not necessary; the continuity bound still applies 
        break
    end
    
end
optval=worstval;

end

function [newverts errlist] = replacevert(coords)
% Given XZ coordinates of a point with x>0 and outside the unit circle, returns two output points, with the following property: each output point lies on one of the tangents to the circle that pass through the input point, and the line segment between the two output points is tangent to the circle.
% This procedure is justified by an inductive argument: Suppose we have a convex polygon that circumscribes a circle, and consider any vertex. It connects to exactly two edges, which are tangents to the circle, matching the scenario outlined in the above procedure. Replacing this vertex with two new vertices as described above results in a new polygon that is still convex and still circumscribes the circle.
% The two output points are ordered by decreasing Z-coordinate (= increasing inclination angle).
rinput = norm(coords); thinput = acos(coords(2)/rinput); %Matlab uses convention acos:[-1,1]->[0,pi], so thinput is the inclination (spherical-coordinate polar angle)
deltth = atan(sqrt((rinput-1)/(rinput+1)));
rnew = sqrt(1 + (rinput-1)/(rinput+1)); thsnew = [thinput-deltth thinput+deltth];
errlist = []; 
if norm(rnew)<1 || thsnew(1)<0 || thsnew(2)>pi
    errlist = [errlist; rnew thsnew];
end
newverts = rnew*[sin(thsnew(1)) cos(thsnew(1)); sin(thsnew(2)) cos(thsnew(2))];
end

function [bestcertval bestfeasval bestcertsol bestfeassol certval feasval] = minFW(thA,vecB,corrs,lambdas,pNPP,pA0,perturb,difftolFW,seeddat,breakval,maxitFW)
% FW bound on the minimum entropy for fixed measurements (as specified by thA and vecB). 
% This can potentially return values >1 in some cases. This is not a mistake - it means that the provided value of gamma cannot be attained with the angles th, hence the constrained optimization takes value +infty and this is a valid lower bound on it.
% Terminates when the first of these occurs: (1) number of iterations > maxits, (2) gap between current best feasible and certified values are within difftol of each other, (3) certified value is larger than breakval.
% Returns the largest certified value, smallest feasible value, the states yielding those values, and (perhaps not very useful) the certified value and feasible value at the iteration where it terminated (note that at least one of these must be equal to the respective best value).
% seeddat is intended to allow "continuing" the algorithm from a previous result. It is either [] or a 1x6 array - if [], a heuristic seed point is used for the FW algorithm, otherwise seedpt specifies a feasible value of the optimization followed by a seed point to start the FW algorithm from.
if isempty(seeddat)
    numtrials=1; [feasval, tup0] = heudualfuncth(thA,vecB,corrs,lambdas,pNPP,pA0,perturb,numtrials); % Generates heuristic starting point. Optimization is convex (no local minima) so one trial should suffice (in any case, this first heuristic trial is merely to generate a starting point)
    bestfeasval=feasval;
else
    bestfeasval=seeddat(1); tup0=seeddat(2:6);
    feasval=dualobj(tup0,thA,vecB,corrs,lambdas,pNPP,pA0,perturb);
end
tupk=tup0; 
bestfeassol = tupk; bestcertval = -1e5;
for iter=1:maxitFW 
    
    rhok=makerho(tupk);
    yalmip('clear')
    delttup=sdpvar(1,5); deltwts=[delttup(1:3) -sum(delttup(1:3))]; % This ensures deltrho has trace 0 and hence rhok+deltrho has trace 1. The changes to the offdiag terms will be delttup(4:5).
    unnormbell = [1 0 0 1; 0 1 -1 0; 1 0 0 -1; 0 1 1 0]'; deltrho = zeros(4,4);
    for j=1:4
        deltrho = deltrho + deltwts(j)*unnormbell(:,j)*unnormbell(:,j)'/2;
    end
    deltrho = deltrho + delttup(4)*(unnormbell(:,1)*unnormbell(:,2)' + unnormbell(:,2)*unnormbell(:,1)')/2 + delttup(5)*(unnormbell(:,3)*unnormbell(:,4)' + unnormbell(:,4)*unnormbell(:,3)')/2;

    % For brevity, the fobj used here for the SDP does not include the lambdas.corrs term (this does not affect how much the tangent function can change over the domain).
    fobj = affbound(deltrho,rhok,[0 thA],pNPP,pA0,perturb) - trace(deltrho*oplG(lambdas,thA,vecB));
    fobj = (fobj+fobj')/2; %Numerical issues sometimes produce small imaginary terms
    constr = [rhok+deltrho>=0];
    options=sdpsettings('verbose',0,'solver','mosek','cachesolvers',1); %cachesolvers value is mainly for use on the cluster
    optimize(constr,fobj,options);
    optlindelt = value(fobj); %This is the maximum amount the tangent function (at rhok) can decrease on the domain (it must be a negative value).
    
    certval = feasval+optlindelt; certvals(iter) = certval;
    if feasval < bestfeasval
        bestfeasval = feasval;
        bestfeassol = tupk;
    end
    if certval > bestcertval
        bestcertval = certval;
        bestcertsol = tupk;
    end
    bestgap = bestfeasval - bestcertval;
    if bestgap < difftolFW || certval > breakval %Checks termination conditions
        break
    end
    optdelt = value(delttup);
    
    fminobj=@(scal)real(dualobj(tupk+scal*optdelt,thA,vecB,corrs,lambdas,pNPP,pA0,perturb));
    vecsize=1; numtrials=1; %Again, optimization is convex so numtrials=1 should suffice
    sols = zeros(numtrials,vecsize); vals = 100*ones(numtrials,1);
    for trial=1:numtrials
        scal0 = 0; 
        Aineq = []; bineq = [];
        Aeq = []; beq = [];
        lb = [0];
        ub = [1];
        options = optimset('Display','off');
        warning('off','MATLAB:nearlySingularMatrix') %Suppresses warnings about near-singular matrices, since we only need this for heuristic purposes anyway
        [sols(trial,:),vals(trial)] = fmincon(fminobj,scal0,Aineq,bineq,Aeq,beq,lb,ub,[],options);
        warning('on','MATLAB:nearlySingularMatrix')
    end
    [feasval, optvalpos] = min(vals);
    optscal = sols(optvalpos,:);
    tupk = tupk+optscal*optdelt; %This is the state corresponding to the current feasval
    
%     % Displays output of every iteration (except the last one, which breaks before reaching this point unless it's at iteration number maxits). Comment out to suppress output.
%     if iter<maxitFW %Not strictly necessary, but this avoids printing the line twice if maxits is reached.
%         temp=[iter bestgap optlindelt feasval tupk]; fprintf(['FW iter %3.d:' repmat(' %9.10f',1,numel(temp)-1) '\n'],temp)
%     end
    
end %iter loop

% % Displays output of last iteration. Comment out to suppress output.
% temp=[iter bestgap optlindelt feasval tupk]; fprintf(['FW iter %3.d:' repmat(' %9.10f',1,numel(temp)-1) '\n'],temp)

% % Optional: this recomputes the best certified value *without* assuming the almost-Bell-diagonal structure, if we want a double-check
% yalmip('clear')
% rho=sdpvar(4,4,'symmetric','real');
% fobj = affbound(rho,makerho(bestcertsol),[0 thA],pNPP,pA0,perturb) - trace(rho*oplG(lambdas,thA,vecB)) + lambdas*corrs';
% fobj = (fobj+fobj')/2;
% constr = [rho>=0, trace(rho)==1];
% options=sdpsettings('verbose',0,'solver','mosek');
% optimize(constr,fobj,options);
% checkcertval = value(fobj);
% if abs(bestcertval-checkcertval) > difftol/1e-3
%     fprintf(['Warning: Value without assumption = %9.9f, value with assumption = %9.9f \n'],[checkcertval bestcertval])
% end
% bestcertval=checkcertval;

end

function [optval optsol] = threshdualfuncth(thresh,maxtries,thA,vecB,corrs,lambdas,pNPP,pA0,perturb,numtrials)
% Loops over heudualfuncth until a value <thresh is found, up to a maximum of maxtries
sols=zeros(maxtries,5); vals=100*ones(maxtries,1);
for trial=1:maxtries
    [vals(trial), sols(trial,:)] = heudualfuncth(thA,vecB,corrs,lambdas,pNPP,pA0,perturb,numtrials);
    if vals(trial) < thresh
        break
    end
end
[optval, optvalpos] = min(vals);
optsol = sols(optvalpos,:);
end

function [optval optsol] = heudualfuncth(thA,vecB,corrs,lambdas,pNPP,pA0,perturb,numtrials)
% Heuristic bound on the minimum entropy for fixed measurements (as specified by thA and vecB). 
% This can potentially return values >1 in some cases. This is not a mistake - it means that the provided correlations cannot be attained with the specified measurements, hence the constrained optimization takes value +infty and this is a valid lower bound on it.
% numtrials specifies the number of attempts (the smallest value over all attempts is returned).
% Since the optimization is convex, typically a small value of numtrial suffices - there are no local minima to be trapped in.
fminobj=@(tup)real(dualobj(tup,thA,vecB,corrs,lambdas,pNPP,pA0,perturb));
vecsize=5; sols = zeros(numtrials,vecsize); vals = 100*ones(numtrials,1);
for trial=1:numtrials
    tup0 = rand(1,5)/4; 
    Aineq = [1 1 1 0 0]; bineq = [1]; 
    Aeq = []; beq = [];
    lb = [0 0 0 -2 -2];
    ub = [1 1 1 2 2];
    options = optimoptions(@fmincon,'Algorithm','sqp','Display','off','OptimalityTolerance',1e-12,'StepTolerance',1e-12,'ConstraintTolerance',1e-12,'MaxFunctionEvaluations',10^4);
    [sols(trial,:),vals(trial)] = fmincon(fminobj,tup0,Aineq,bineq,Aeq,beq,lb,ub,@negmineig,options);
end
[optval, optvalpos] = min(vals);
optsol = sols(optvalpos,:);
end

function affbound = affbound(rho,rhotan,thAs,pNPP,pA0,perturbthresh)
% Affine lower bound on the relative-entropy terms at the point rhotan. (Does not include Lagrange multiplier terms.)
pauli = {[0 1; 1 0], [0 -i; i 0], [1 0; 0 -1]};
pinputs = [pA0 1-pA0]; affbound=0;
for x=1:2
    rhoTAB = cflip(rho,pNPP(x)); rhotanTAB = cflip(rhotan,pNPP(x));
    perturb = max([perturbthresh, 1.5*8*-min(eig(rhotanTAB))]);
    rhotanTAB=(1-perturb)*rhotanTAB+perturb*eye(8)/8;

    ZrhotanTAB = zeros(8,8); 
    for a=-1:2:1
        pinchop = otimes({eye(2),(eye(2)+a*(cos(thAs(x))*pauli{3}+sin(thAs(x))*pauli{1}))/2,eye(2)}); %No need to worry about the 0/1 to +/- mapping here, because it just sums over both terms anyway
        ZrhotanTAB = ZrhotanTAB + pinchop*rhotanTAB*pinchop;
    end
    affbound=affbound + pinputs(x)*trace(rhoTAB*( logm(rhotanTAB)-logm(ZrhotanTAB) ))/log(2);
end
end

function dualobj = dualobj(tup,thA,vecB,corrs,lambdas,pNPP,pA0,perturb)
% tup is to be in the form [wt1 wt2 wt3 offdiag1 offdiag2] describing the almost-Bell-diagonal rho_AB.
% Returns a large positive "penalty value" if input parameters do not yield a PSD state. In theory, this should mean we could omit the nonlcon argument from fmincon, but in practice it seems to perform much better when that argument is properly supplied.
pauli = {[0 1; 1 0], [0 -i; i 0], [1 0; 0 -1]};
[negmineigval notused] = negmineig(tup); 
if negmineigval>0 %Checking for negative eigenvalues
    dualobj = 10 + 10*negmineigval; %Imposing a penalty function in the negative-eigenvalue regime
else
    thAs=[0 thA]; pinputs = [pA0 1-pA0]; dualobj=0; 
    for x=1:2
        rhoTAB = cflip(makerho(tup),pNPP(x)); rhoTAB=(1-perturb)*rhoTAB+perturb*eye(8)/8;
        ZrhoTAB=zeros(8,8); 
        for a=-1:2:1
            pinchop = otimes({eye(2),(eye(2)+a*(cos(thAs(x))*pauli{3}+sin(thAs(x))*pauli{1}))/2,eye(2)}); %No need to worry about the 0/1 to +/- mapping here, because it just sums over both terms anyway
            ZrhoTAB = ZrhoTAB + pinchop*rhoTAB*pinchop; 
        end
        dualobj = dualobj + pinputs(x)*divKL(rhoTAB,ZrhoTAB);
    end
    dualobj = dualobj - trace(makerho(tup)*oplG(lambdas,thA,vecB)) + lambdas*corrs';
end
end

function dualobj2 = dualobj2(tup,coeffs,lambda,gamma,pNPP,pA0,perturb)
% tup is to be in the form [wt1 wt2 wt3 thA0 thA1 thB0 thB1]
pauli = {[0 1; 1 0], [0 -i; i 0], [1 0; 0 -1]};
wts = [tup(1:3) 1-sum(tup(1:3))]; thAs=tup(4:5); thBs=tup(6:7); 
if min(wts)<0 %Checking for negative eigenvalues - fmincon does not always obey the Aineq, bineq constraints.
    dualobj2 = 10 + 10*-min(wts); %Imposing a penalty function in the negative-eigenvalue regime
else
    unnormbell = [1 0 0 1; 0 1 -1 0; 1 0 0 -1; 0 1 1 0]'; rhoAB = zeros(4,4);
    for j=1:4
        rhoAB = rhoAB + wts(j)*unnormbell(:,j)*unnormbell(:,j)'/2;
    end
    pinputs = [pA0 1-pA0]; dualobj2=0; 
    for x=1:2
        rhoTAB = cflip(rhoAB,pNPP(x)); rhoTAB=(1-perturb)*rhoTAB+perturb*eye(8)/8;
        ZrhoTAB=zeros(8,8); 
        for a=-1:2:1
            pinchop = otimes({eye(2),(eye(2)+a*(cos(thAs(x))*pauli{3}+sin(thAs(x))*pauli{1}))/2,eye(2)}); %No need to worry about the 0/1 to +/- mapping here, because it just sums over both terms anyway
            ZrhoTAB = ZrhoTAB + pinchop*rhoTAB*pinchop; 
        end
        dualobj2 = dualobj2 + pinputs(x)*divKL(rhoTAB,ZrhoTAB);
    end
    dualobj2 = dualobj2 - lambda*(bellths(coeffs,rhoAB,thAs,thBs)-gamma);
end
end

function cflip = cflip(rho,noisify)
% Computes the "coherently flipped" version of a two-qubit state on AB and outputs the state on TAB where T is the flipping ancilla
ancilla=[sqrt(1-noisify) sqrt(noisify)]'; ancilla=ancilla*ancilla';
cflipop = otimes({diag([1 0]),eye(4)}) + otimes({diag([0 1]),[0 -i; i 0],eye(2)});
cflip = cflipop*otimes({ancilla,rho})*cflipop; %No adjoint required on last term because the unitary cflipop is also Hermitian in this case
end

function [negmineig notused] = negmineig(tup)
% Computes negative of minimum eigenvalue of makerho(tup), for use in fmincon. The second argument in the output is a just a trivial placeholder for compatibility with fmincon.
% Could in principle simply be computed using eig(), but for such states we can derive a closed-form expression, which should be slightly faster.
wts = [tup(1:3) 1-sum(tup(1:3))];
negmineig = -min([wts(1)+wts(2)-sqrt((wts(1)-wts(2))^2+4*tup(4)^2) wts(3)+wts(4)-sqrt((wts(3)-wts(4))^2+4*tup(5)^2)]);
notused = [];
end

function rho = makerho(tup)
% WARNING: does not check the result is PSD. tup is to be a tuple in the form [wt1 wt2 wt3 offdiag1 offdiag2]. 
wts = [tup(1:3) 1-sum(tup(1:3))]; 
unnormbell = [1 0 0 1; 0 1 -1 0; 1 0 0 -1; 0 1 1 0]'; rho = zeros(4,4);
for j=1:4
    rho = rho + wts(j)*unnormbell(:,j)*unnormbell(:,j)'/2;
end
rho = rho + tup(4)*(unnormbell(:,1)*unnormbell(:,2)' + unnormbell(:,2)*unnormbell(:,1)')/2 + tup(5)*(unnormbell(:,3)*unnormbell(:,4)' + unnormbell(:,4)*unnormbell(:,3)')/2;
end

function oplG = oplG(lambdas,thA,vecB)
pauli = {[0 1; 1 0], [0 -i; i 0], [1 0; 0 -1]};
sum1 = lambdas(1)*pauli{3} + lambdas(3)*(cos(thA)*pauli{3}+sin(thA)*pauli{1});
sum2 = lambdas(2)*pauli{3} + lambdas(4)*(cos(thA)*pauli{3}+sin(thA)*pauli{1});
oplG = otimes({sum1,pauli{3}}) + otimes({sum2,vecB(2)*pauli{3}+vecB(1)*pauli{1}});
end

function bellths = bellths(coeffs,rho,thAs,thBs)
bellths=0;
for x=1:2
for y=1:2
projs=angs2projsxy([thAs(x),0,thBs(y),0]);
    for a=1:2
    for b=1:2
        bellths = bellths + coeffs(a,b,x,y)*trace(rho*otimes({projs{1,a},projs{2,b}}));
    end
    end
end
end
end

function coeffscorr2 = coeffscorr2(x,y)
coeffscorr2 = zeros(1,2^4); coeffscorr2 = reshape(coeffscorr2, [2 2 2 2]);
for a = 1:2
    for b = 1:2
        coeffscorr2(a,b,x,y) = 2*(a-1.5) * 2*(b-1.5); %For these terms the +- labelling doesn't matter as long as the same convention is used for both Alice and Bob
    end
end
end

function makeprobsxy = makeprobsxy(rho,projs)
% Probs are indexed as [a,b]
makeprobsxy = zeros(2,2); 
for a=1:2
for b=1:2
    makeprobsxy(a,b) = trace(rho*otimes({projs{1,a},projs{2,b}}));
end
end
end

function angs2projsxy = angs2projsxy(angs)
% Projectors are indexed as {Alice/Bob,output}
% angs is read as [tha,pha,thb,phb]
angs2projsxy = cell(2,2);
for j=0:1
    if j==0
        th=angs(1);phi=angs(2);
    else
        th=angs(3);phi=angs(4);
    end
    eig = [cos(th/2);exp(i*phi)*sin(th/2)];
    angs2projsxy{j+1,1} = eig*eig';
    angs2projsxy{j+1,2} = eye(2)-eig*eig';
end
end

function otimes = otimes(matrices)
otimes = kron(matrices{1},matrices{2});
for n = 3:length(matrices)
    otimes = kron(otimes,matrices{n});
end
end

function ket = ket(bitlist)
basis = {[1; 0], [0; 1]};
qubits = cell(length(bitlist),1);
for n = 1:length(bitlist)
    qubits{n} = basis{bitlist(n) + 1};
end
ket = otimes(qubits);
end

function divKL = divKL(rho1,rho2)
divKL = trace(rho1*(logm(rho1)-logm(rho2)))/log(2);
end

function shannon = shannon(probs)
shannon = 0;
for j=1:length(probs)
    shannon = shannon - probs(j)*log2(probs(j));
end
end