% Funzione per il calcolo delle feature distintive per FOGLIE/UNKNOWN
% Input: maschera binaria dell'immagine, immagine RGB
% Output: features_value = matrice con valori dei descrittori
function features_value = unknown_feature_extractor(image_mask, img_rgb)
%% Labelizzazione
[A, nlabels] = bwlabel(image_mask);

%% STATS di tutti gli oggetti nell'immagine
stats = regionprops(A,'Eccentricity', 'Circularity', 'Area', 'Solidity');
for i=1:nlabels
    obj_mask = A ==i;
    obj_rgb = img_rgb .* uint8(repmat(obj_mask, [1, 1, size(img_rgb, 3)]));
    obj_bw = rgb2gray(obj_rgb);
    var_gray = var(double(obj_bw(:)));

    %% Calcolo MBB (minimum bounding box)
    [~, areaMBB] = calculate_mbb(obj_mask);
    rate = stats(i).Area/areaMBB;
     
    %% Assegno features
    % Questo da ottimi risultati, ma non è abbastanza robusto su altri
    % dataset (dove per esempio ci sono oggetti uniformi con forme poche
    % forme geometriche)
    %%features_value(i, :) = [rate, var_gray];
    
    % Più features che rendono il modello piu robusto, anche ad oggetti
    % "simili" a foglie
    features_value(i, :) = [rate, stats(i).Circularity, stats(i).Eccentricity, var_gray];

end