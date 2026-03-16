% Funzione per il calcolo delle feature delle foglie, unknown esclusi
% Input: maschera binaria dell'immagine, immagine RGB, predizioni del
% modello FOGLIE/UNKNOWN del test (per i dati di training, pred = [])
% Output: pred = array delle predizioni aggiornato per la gestione del numero di oggetti
% variabile per foto, features_value = matrice con valori dei descritttori
function [pred, features_value] = leaves_feature_extractor(image_mask, img_rgb, pred)
%% Labelizzazione
[A, nlabels] = bwlabel(image_mask);

%% Non calcolo features degli unknown
if ~isempty(pred)
    for i=1:nlabels
        if~strcmp(pred(i), 'Foglia')
            A(A == i) = 0;
        end
    end
    % % Rimuove i primi nlabels elementi
    pred(1:nlabels) = [];
end

% Ricalcolo label senza UNKNOWN
[A, nlabels] = bwlabel(A);


%% STATS di tutti gli oggetti nell'immagine
stats = regionprops(A, 'Perimeter', 'Eccentricity', 'Circularity', 'Solidity');

%% Estrazione delle features
for i=1:nlabels
    obj_mask = A ==i;
    obj_rgb = img_rgb .* uint8(repmat(obj_mask, [1, 1, size(img_rgb, 3)]));
    obj_bw = rgb2gray(obj_rgb);
    obj_double = double(obj_bw);
    %%  Estraggo primo e secondo momento di Hu
    [height, width] = size(obj_double);

    xgrid = repmat((-floor(height/2):1:ceil(height/2)-1)',1,width);
    ygrid = repmat(-floor(width/2):1:ceil(width/2)-1,height,1);

    [x_bar, y_bar] = centerOfMass(obj_double,xgrid,ygrid);

    xnorm = x_bar - xgrid;
    ynorm = y_bar - ygrid;

    mu_11 = central_moments(obj_double, xnorm, ynorm, 1,1);
    mu_20 = central_moments(obj_double , xnorm, ynorm, 2,0);
    mu_02 = central_moments(obj_double ,xnorm, ynorm, 0,2);

    I_one   = mu_20 + mu_02;
    I_two   = (mu_20 - mu_02)^2 + 4*(mu_11)^2;

    %% Assegno features
    features_value(i, :) = [stats(i).Circularity, stats(i).Eccentricity, stats(i).Solidity, stats(i).Perimeter, I_one, I_two];
end
end

% Alcune features scartate:
    % features_value(i, :) = [stats(i).Circularity, stats(i).Eccentricity,
    % stats(i).Solidity, stats(i).Perimeter) => poco robusto
    % 7 momenti di Hu + precedenti => rischio di overfitting
    % Altro: Entropy, Skewness G, Kurtosis G, Fractal Dimension, Symmetry
    % Score, Shape Irregularity Score (SIS), Correlation, Fourier description (1-10)
    % e Momenti di Zernike
