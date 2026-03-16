% Funzione per la ridefinizione dettagliata di una maschera etichettata
% a partire da un'immagine in input e maschera precedente
% Input: bw = immagine bianca e nera, k_mask = maschera in output di
% localizer()
% Output: KMask: maschera dettagliata, dove ogni = oggetto nella
% foto
function newKMask = localizer_with_edges(bw, k_mask)
load_constants
%% Pre processing
sharp_img = imsharpen(bw, Radius=2,Amount=0.8);

%% Segmentazione
sobel = edge(sharp_img, "sobel");

% Creazione di un elemento strutturante
se = strel('disk',DISK_SIZE); 

% Riempio ed erodo bordi
connectedEdges = imfill(imdilate(sobel, se), 'holes');
connectedEdges = imerode(connectedEdges, se);

% Rimozione del rumore con filtro mediano
medd = medfilt2(connectedEdges, [MED_SIZE MED_SIZE]);
cleanEdges = bwareaopen(medd, PIXEL_THRESHOLD);


% Labelizzazione della nuova maschera
[EdgeMask, numLabelsEdge] = bwlabel(cleanEdges); % Etichettatura delle regioni connesse

%% Gestione di oggetti che nella maschera k_mask non sono stati binarizzati correttamente
% (in particolare oggetti con tanti dettagli)
for i = 1:numLabelsEdge
    obj = (EdgeMask == i); 

    % Conto pixels dell'oggetto
    edge_label_area = nnz(obj);

    % Machera pixel in comune
    comm_pixels = k_mask & obj;

    % Conta i pixel in comune
    n_comm_pixels = nnz(comm_pixels); 

    % Calcolo rate pixel in comune/pixel dell'oggetto
    ratio = n_comm_pixels/edge_label_area;
    
    % Controllo il valore del rate dell'area, se è molto basso
    % unisco (OR) con la maschera precedente
    if round(ratio, 2) <= PIXEL_RATIO_THRESHOLD % 
        k_mask = k_mask | obj;
    end
end


% Ricalcolo labels corrette per segmentazione finale
[KMask, nlabels] = bwlabel(k_mask);

% Rimuovo le label con area molto piccola (ulteriore prevenzione di errori)
stats = regionprops(k_mask, 'Area');
areas = [stats.Area];
newKMask = KMask;
for i = 1:nlabels
    if areas(i) < MIN_AREA
        % Rimuovi l'etichetta corrente dalla maschera (azzera quei pixel)
        newKMask(newKMask == i) = 0;
    end
end

end