% Funzione per l'annotazione delle immagini finali
function [labels, contorni] = annotate_images(img_mask, mask_gt, img, labels)
% Estrai i bordi dalla maschera binaria
contorni = bwperim(img_mask);

[labeled_img, num_objects] = bwlabel(img_mask);

stats = regionprops(labeled_img, 'Centroid', 'BoundingBox');

% Calcolo del coefficiente di Jaccard
similarity = "Coefficiente di Jaccard: " + string(round((jaccard(logical(img_mask), mask_gt)),2));

% Aggiungi le etichette per ogni oggetto
for i = 1:num_objects
    bbox = stats(i).BoundingBox;
    img = insertObjectAnnotation(img,"rectangle",bbox, labels(i), 'Color', 'yellow', 'FontSize', 18, 'LineWidth', 7);
end

% Mostra l'immagine con le annotazioni
figure;
imshow(img); title(similarity)
hold on;

% Aggiungo i contorni
[y, x] = find(contorni);
plot(x, y, 'b.', 'MarkerSize', 3);
hold off;

% Tolgo i primi oggetti dalla lista stampati (per gestire
% indicizzazione)
labels(1:num_objects) = [];
end

