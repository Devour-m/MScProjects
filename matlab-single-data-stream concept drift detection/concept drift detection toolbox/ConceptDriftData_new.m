function data = ConceptDriftData_new(type,N)
  % switch on the type of experiment
  switch type
    case 'checkerboard'
        T = 8; % number of concept drift
        alpha = linspace(0,2*pi,T);
        a = 0.5;
        X = [];
        y = [];
        for t = 1:T
            [concept_new_X,concept_new_y] = gendatcb(N,a,alpha(t));
            X = [X;concept_new_X];
            y = [y;concept_new_y];
        end
        data = [X,y];
      
      case 'sea'
          T = 8;
          theta = [2,5];
          theta = [theta,theta,theta,theta];
          noise = 0.05;
          X = [];
          y = [];
          for t = 1:T
              [concept_new_X,concept_new_y] = SEA(theta(t),N,noise);
              X = [X;concept_new_X'];
              y = [y;concept_new_y'];
          end
          data = [X,y];

    otherwise
      error('ERROR::conceptDriftData_new.m: Unknow Dataset Selected');
  end
end

function [X,Y] = SEA(theta,N,noise)
  Nnoise = floor(N*noise);
  %%%% generate data and labels associated with each instance
  x = 10*rand(3,N);
  X = x; % save data in X
  Y = zeros(1,length(X));
  x(3,:) = []; % remove feature w/ no information
  Y(sum(x)>=theta) = 1;
  Y(Y~=1) = 2;
  %%%% add noise into the dataset
  r = randperm(numel(Y));
  r(Nnoise+1:numel(Y)) = [];
  Y(r) = Y(r)-1; % change 2->1 and 1->0
  Y(Y==0) = 2;   % class '0' is actually class '2' with noise
  Y(find(Y==2))=0;
end

function [d,labd] = gendatcb(N,a,alpha)
% N data points, uniform distribution,
% checkerboard with side a, rotated at alpha
d = rand(N,2);
d_transformed = [d(:,1)*cos(alpha)-d(:,2)*sin(alpha), ...
    d(:,1)*sin(alpha)+d(:,2)*cos(alpha)];
s = ceil(d_transformed(:,1)/a)+floor(d_transformed(:,2)/a);
labd = 2-mod(s,2);
labd(find(labd==2))=0;
end

%{
function data = ConceptDriftData_new(type,N)
  % switch on the type of experiment
  switch type
    case 'checkerboard'
         T = 5; % number of concept drift
        % alpha = linspace(0,2*pi,T);
         alpha = [pi/6, 2*pi/6,3*pi/6, 4*pi/6,5*pi/6];
        a = 0.5;
        X = [];
        y = [];
        for t = 1:T
            [concept_new_X,concept_new_y] = gendatcb(N,a,alpha(t));
            X = [X;concept_new_X];
            y = [y;concept_new_y];
        end
        data = [X,y];
      
      case 'sea'
          T = 8;
          theta = [2,5];
          theta = [theta,theta,theta,theta];
          noise = 0.05;
          X = [];
          y = [];
          for t = 1:T
              [concept_new_X,concept_new_y] = SEA(theta(t),N,noise);
              X = [X;concept_new_X'];
              y = [y;concept_new_y'];
          end
          data = [X,y];

    otherwise
      error('ERROR::conceptDriftData_new.m: Unknow Dataset Selected');
  end
end

function [X,Y] = SEA(theta,N,noise)
  Nnoise = floor(N*noise);
  %%%% generate data and labels associated with each instance
  x = 10*rand(3,N);
  X = x; % save data in X
  Y = zeros(1,length(X));
  x(3,:) = []; % remove feature w/ no information
  Y(sum(x)>=theta) = 1;
  Y(Y~=1) = 2;
  %%%% add noise into the dataset
  r = randperm(numel(Y));
  r(Nnoise+1:numel(Y)) = [];
  Y(r) = Y(r)-1; % change 2->1 and 1->0
  Y(Y==0) = 2;   % class '0' is actually class '2' with noise
  Y(find(Y==2))=0;
end

function [d,labd] = gendatcb(N,a,alpha)
% N data points, uniform distribution,
% checkerboard with side a, rotated at alpha
d = rand(N,2);
d_transformed = [d(:,1)*cos(alpha)-d(:,2)*sin(alpha), ...
    d(:,1)*sin(alpha)+d(:,2)*cos(alpha)];
s = ceil(d_transformed(:,1)/a)+floor(d_transformed(:,2)/a);
labd = 2-mod(s,2);
labd(find(labd==2))=0;
end
%}