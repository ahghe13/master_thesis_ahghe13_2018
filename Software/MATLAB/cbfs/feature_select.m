% Master's thesis
% File description: Based on the input correlation matrix, the
%					function selects the least correlated features
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

function [features] = feature_select(corr_matrix, no_of_features)
	[rows, cols] = size(corr_matrix);

	features = 1:cols;

	corr_matrix = corr_matrix - diag(diag(corr_matrix));

	while size(features,2) ~= no_of_features
		[M, I] = max(corr_matrix);
		[M, R] = max(M);
		C = I(R);
		D = C;
		if sum(corr_matrix(R,:)) > sum(corr_matrix(C,:))
			D = R;
		end
		corr_matrix(D,:) = 0;
		corr_matrix(:,D) = 0;
		I = find(features == D);

		features(I) = [];
	end
	features = transpose(features);
end