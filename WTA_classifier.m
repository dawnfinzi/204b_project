function WTA_classifier(data_path, results_path, session, condition, ROIs)
%
%    Usage: Runs a winner-take-all correlation classifier
%    Adapted by Dawn Finzi from a script written by Marisa Nordt & Mareike Grotheer
%    Date: 06/06/2018
%    Notes: This is currently specified for the dynamic/static localizer
%    experiment. The within function parameters (lines 46 to 50) will need
%    to be adapted if used for analyzing a different experiment. 
%    
%    Inputs:
%       - data_path --> string with the general data path
%       - results_path --> string with the path where you want the results
%       to be stored
%       - session --> string containing the name of the session folder
%       - condition --> string containing which condition to analyze (i.e.
%       static or dynamic)
%       - ROI --> string containing the ROIs to use for classification
%
%    Output: classification results for the category and superordinate
%    category classifiers and confusion matrices for the category
%    classifier are saved as .mat files (with ROI and condition in the name)
%    in the specified results path
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example Input: 
% % Set the session and the condition
% session = {'kgs072010_new'};
% condition = 'static';
%
% % Set the paths
% data_path = '/sni-storage/kalanit/biac2/kgs/projects/Dawn/Psych204B/data';
% results_path = '/sni-storage/kalanit/biac2/kgs/projects/Dawn/Psych204B/results';
% 
% % Specify the ROIs 
% ROIs={'rh_V1'};
%
% % Call the function
% WTA_classifier(data_path, results_path, session, condition, ROIs)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set up 
% Choose which runs to use depending on the condition
if strcmp(condition,'static')
    listRuns = repmat([2 4 6], 1, 1);
elseif strcmp(condition,'dynamic')
    listRuns = repmat([1 3 5], 1, 1);
end

% Set experiment parameters.
dataType= 3;
numconds = 6; %faces emo, faces comm, bodies emo, hand comm, vehicles, scenes
SO_cats = 4; %superordinate categories (faces/limbs/vehicles/scenes)
eventsPerBlock = 6; %TRs per block
nrRuns = 3;
s = 1; %we only have one subject in this experiment

% Preallocate variables.
correctC=nan(numconds, nrRuns);
conf=zeros(numconds,numconds, nrRuns);
correctSO=zeros(SO_cats,nrRuns);

% Let's make sure the specified data exists in the right location
assert(exist(fullfile(data_path,session{1}))~=0);
pathname=fullfile(data_path,session{1});
eval(['cd ' pathname]); %navigate to the specific data directory

%% Okay let's do it
for r=1:length(ROIs) %for each ROI
    
    ROIname=ROIs{r};

    if r ==1 %initialize a hidden inplane on the first iteration
       hi=initHiddenInplane(dataType, 1, ROIname);
       currentview = 'inplane';
    end

    % Initialize mvs
    mv1=mv_init(hi,ROIname,listRuns(s,1),dataType);
    mv2=mv_init(hi,ROIname,listRuns(s,2),dataType); 
    mv3=mv_init(hi,ROIname,listRuns(s,3),dataType);

    mvs = {mv1, mv2, mv3};

    % Set parameters in each mv.
    for m=1:nrRuns
         params = mvs{m}.params;
         params.eventAnalysis = 1;
         params.detrend = 1; % high-pass filter (remove the low frequency trend)
         params.detrendFrames = 20;
         params.inhomoCorrection = 1; % divide by mean (transforms from raw scanner units to % signal)
         params.temporalNormalization = 0; % no matching 1st temporal frame
         params.ampType = 'betas';
         params.glmHRF = 3; % SPM hrf
         params.eventsPerBlock = eventsPerBlock;
         params.lowPassFilter = 0; % no temporal low pass filter of the data

         if m==1
            mv1.params = params;
         elseif m==2
            mv2.params = params;
         elseif m==3
            mv3.params = params;
         end
    end

    % Apply GLM. 
    mv(1) = mv_applyGlm(mv1);
    mv(2) = mv_applyGlm(mv2);
    mv(3) = mv_applyGlm(mv3);

    % Get betas for each run and voxel - does not compute amps for null
    % condition
    amps(1,:,:)=mv_amps(mv1); %shape: voxels x conditions
    amps(2,:,:)=mv_amps(mv2);
    amps(3,:,:)=mv_amps(mv3);
    
    for i = 1:nrRuns
         % subtract means from each voxel
         mean_for_run(:,i) = squeeze(mean(amps(i,:,:),3));
         meanmat(:,:,i) = mean_for_run(:,i)*ones(1,numconds);
         subbetas(:,:,i) = squeeze(amps(i,:,:))-squeeze(meanmat(:,:,i));
         
         % For z-values we also need the residual variance and the degrees of
         % freedom. z-values = subtracted-beta value devided by the standard deviation which is defined as
         % sqrt( residual ^2 / [degrees of freedom] )
         residualV(:,:,i) = mv(i).glm.residual; 
         dof(i) = mv(i).glm.dof;
         residualvar(i,:,:) = sum(residualV(:,:,i).^2)/dof(i);
         resd(i,:,:) = sqrt(residualvar(i,:,:));
         resd_T = resd(i,:,:)';
         resd_mat(i,:,:) = repmat(resd_T, [1 numconds]);
         z_values(i,:,:) = squeeze(subbetas(:,:,i))./squeeze(resd_mat(i,:,:));
    end
    z_values1 = squeeze(z_values(1,:,:)); z_values2 = squeeze(z_values(2,:,:)); z_values3 = squeeze(z_values(3,:,:)); 

    %% Compute Crosscorrelation matrices.
    c12=zeros(numconds,numconds);
    c13=zeros(numconds,numconds);
    c23=zeros(numconds,numconds);

    for i=1:numconds
        for j=1:numconds

            tmpc=corrcoef(z_values1(:,i),z_values2(:,j));
            c12(i,j)=tmpc(1,2);

            tmpc=corrcoef(z_values1(:,i),z_values3(:,j));
            c13(i,j)=tmpc(1,2);

            tmpc=corrcoef(z_values2(:,i),z_values3(:,j));
            c23(i,j)=tmpc(1,2);
        end
    end

    %% WTA classifier

    crossCorrs = {c12, c13, c23};
    % For both directions within each of our matrices (Run: 1-2,1-3,2-3,2-1,3-1,3-2)
    for c = 1:length(crossCorrs)
        cmat = crossCorrs{c};

    % Each matrix (c12, c13, c23) represents the correlation of each of different conditions
    % with the same conditions between two runs.

    % Check if the correlation between a certain condition in one run 
    % is highest for the same condition in the other run.

        for w = 1:numconds
            corrC = cmat(:,w); % from Run 1 to 2
            corrR = cmat(w,:); % and the other way around
            wtaC = find(corrC==max(corrC)); % find the maximum value
            wtaR = find(corrR==max(corrR));

            % Now check if the maximum is the same as w, basically if
            % the highest value is obtained for the category at hand.
            if wtaC == w && wtaR == w % if both predictions are correct
                correctC(w,c) = 1;
                conf(w, wtaR, c) = 0;
            elseif wtaC == w && wtaR ~= w % first one correct but 2nd incorrect
                correctC(w,c) = 0.5;
                conf(w, wtaR, c) = 0.5;
            elseif wtaC ~= w && wtaR == w % second one correct but 1st incorrect
                correctC(w,c) = 0.5;
                conf(w, wtaC, c) = 0.5;
            else % both are incorrect
                correctC(w, c) = 0;
                if wtaC == wtaR % and both took the same category
                    conf(w, wtaC, c) = 1;
                else
                    conf(w, wtaC,c) = 0.5;
                    conf(w, wtaR, c) = 0.5;
                end
            end

            % Superordinate classifier: Here, we set the classifier to
            % correct, if one out of the two members of that category gets
            % the highest value. We basically group categories together to broader categories
            if w <= 4 %SO for faces and bodies only
                if floor((wtaC+1)/2)==floor((w+1)/2)
                    correctSO(floor((w+1)/2),c)= correctSO(floor((w+1)/2),c)+0.25;
                end
                if floor((wtaR+1)/2)==floor((w+1)/2)
                    correctSO(floor((w+1)/2),c)= correctSO(floor((w+1)/2),c)+0.25;
                end
            else %otherwise judge classification separately (for vehicles and scenes)
                if wtaC == w && wtaR == w % if both predictions are correct
                    correctSO(w-2,c) = 1;
                elseif wtaC == w && wtaR ~= w % first one correct but 2nd incorrect
                    correctSO(w-2,c) = 0.5;
                elseif wtaC ~= w && wtaR == w % second one correct but 1st incorrect
                    correctSO(w-2,c) = 0.5;
                else % both are incorrect
                    correctSO(w-2,c) = 0;
                end
            end

        end
        
        %% Now let's save all our results
        % category classification results
        correctCfilename= sprintf('correctC_%s_%s_%d_Runs_z.mat', ROIname, condition, nrRuns');
        correctCfile= fullfile(results_path, correctCfilename);
        save(correctCfile, 'correctC');

        correctC_avRuns = mean(correctC, 2);
        correctC_avRunsfilename = sprintf('correctC_avRuns_%s_%s_%d_Runs_zstat.mat', ROIname, condition, nrRuns');
        correctC_avRunsfile = fullfile(results_path, correctC_avRunsfilename);
        save(correctC_avRunsfile, 'correctC_avRuns') 
        
        % domain (superordinate category) classification results
        correctSOfilename= sprintf('correctSO_%s_%s_%d_Runs_z.mat', ROIname,  condition, nrRuns');
        correctSOfile= fullfile(results_path, correctSOfilename);
        save(correctSOfile, 'correctSO');

        correctSO_avRuns = mean(correctSO,2);
        correctSO_avRunsfilename= sprintf('correctSO_avRuns_%s_%s_%d_Runs_z.mat', ROIname,  condition, nrRuns');
        correctSO_avRunsfile= fullfile(results_path, correctSO_avRunsfilename);
        save(correctSO_avRunsfile, 'correctSO_avRuns');

        % confusion matrices for the category classification results
        conffilename = sprintf('conf_%s_%s_%d_Runs_z.mat', ROIname,  condition, nrRuns');
        conffile = fullfile(results_path, conffilename);
        save(conffile, 'conf')

    end
end

end
