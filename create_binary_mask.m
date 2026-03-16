function mask = create_binary_mask(annotations, i)
    % Scala per adattarsi alla risoluzione delle immagini originali (4K)
    scale_x = 1920 / 4000;
    scale_y = 1080 / 2252;
    
    % Filtra gli oggetti pertinenti per l'immagine corrente
    oggetti_img = annotations([annotations.image_id] == (i - 1));
    
    % Crea una maschera vuota delle stesse dimensioni
    mask = zeros(1080, 1918); 
    
    for j = 1:numel(oggetti_img)
        % Estrai i punti di ogni poligono
        punti_poligono = oggetti_img(j).segmentation;
        
        % Scala le coordinate del poligono
        x_points = punti_poligono(1:2:end) * scale_x;  % Scala X
        y_points = punti_poligono(2:2:end) * scale_y;  % Scala Y
        
        % Chiudi il poligono aggiungendo il primo punto alla fine
        x_points = [x_points, x_points(1)];
        y_points = [y_points, y_points(1)];
        
        % Inverte le coordinate per le immagini 1 e 5 (causa errore nella
        % annotazione di queste due immagini)        
        if i == 5 || i == 1
            x_points = 1920 - x_points;
            y_points = 1080 - y_points;
        end
        
        % Creo maschera, aggiungendo di volta in volta i poligoni creati
        mask = mask | poly2mask(x_points, y_points, size(mask, 1), size(mask, 2));
    end
end