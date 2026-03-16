% Funzione per la creazione di una maschera etichettata
% a partire da un'immagine in input
% Input: img_rgb = matrice immagine imread(img))
% Output: cleanEdges: maschera definita, dove ogni label = oggetto nella
% foto

function KMask = localizer(img_rgb)
%% READING IMAGE
% Converto in spazi colore
hsv = rgb2hsv(img_rgb);
gray_img = rgb2gray(img_rgb);

% Estraggo canale saturazione
s = hsv(:,:,2);

%% Segmentazione
% Binarizzo il canale saturazione. Poiche lo sfondo è bianco, rimuovo le
% ombre.
binaryMask = imbinarize(s);

% K means 2 cluster
bwcluster = imsegkmeans(gray_img,2);

% Combinazione delle due maschere
noshadow = binaryMask | ~logical(bwcluster ==1);

% applico riduzione del rumore sparso
object_mask = medfilt2(noshadow);

% Riempio i buchi dentro le foglie
object_mask = imfill(object_mask, 'holes');

% Rimuovo piccoli artefatti e rumore
object_mask = imerode(object_mask, strel('disk', 1));

% Questo metodo mi permette di localizzare oggetti
% con tanti dettagli
KMask = localizer_with_edges(gray_img, object_mask);

end