%% Caricamento costanti, GT, modelli
clear;
load_constants;
disp("Caricamento gt");
Gt_train_class = load("GT_train_class.mat").Gt_train_class;
Gt_test_class = load("GT_test_class.mat").Gt_test_class;
Gt_test_unk = load("Gt_test_unk.mat").Gt_test_unk;
Gt_train_unk = load("Gt_train_unk.mat").Gt_train_unk;

% Caricamento dei modelli
model = load("modelloClassi.mat", "modelloClassi2").modelloClassi2.ClassificationKNN;
modelUnk = load("modelloUnknown.mat", "modelloUnknown2").modelloUnknown2.ClassificationTree;

%% Dal train test, estraggo le soglie per dividere FOGLIE da UNKNOWN
%Estraggo feature dal train set - COMMENTO PERCHE MODELLO GIA ALLENATO
% 
% disp("----Estrazioni dati unknown training-----");
% X_train_unk = [];
% for i = 1:N_IMMAGINI_TRAIN
%     %Costruzione del percorso
%     file_name = fullfile(TRAIN_PATH, ['train-', num2str(i), '.jpg']); 
%     img = imread(file_name);
%     mask = localizer(img);    
%     %Estrai le caratteristiche
%     X_train_obj = unknown_feature_extractor(mask, img);
%     disp(file_name);
%     X_train_unk = [X_train_unk; X_train_obj];
% end
%%
disp("----Estrazioni dati unknown testing-----");
X_test_unk = [];
for i = 1:N_IMMAGINI_TEST

    % Costruzione del percorso
    file_name = fullfile(TEST_PATH, ['test-', num2str(i), '.jpg']);
    img = imread(file_name);
    disp(file_name);

    % Estraggo la maschera etichettata
    mask = localizer(img);   

    % Estraggo feature
    X_test_obj = unknown_feature_extractor(mask, img);
    
    % Estrai le caratteristiche
    X_test_unk = [X_test_unk; X_test_obj];
end

%% Predicto gli unknown 
predUnk_test = predict(modelUnk, X_test_unk);
predictions_metrics(string(predUnk_test), Gt_test_unk);

% Calcolo predizioni e metriche
%predUnk_train = predict(modelUnk, X_train_unk);
%predictions_metrics(string(predUnk_train), Gt_train_unk);

%% Estrazione features per ALLENARE MODELLO 10 CLASSI (SOLO FOGLIE, no unk)
% disp("----Estrazioni dati per foglie training-----");
% (non è necessario far partire questa parte di codice perche il modello è
% gia caricato)

% 
% X_train_leaves = [];
% 
% for i = 1:N_IMMAGINI_TRAIN
%     %Costruzione del percorso
%     file_name = fullfile(TRAIN_PATH, ['train-', num2str(i), '.jpg']); 
%     img = imread(file_name);
%     disp(file_name);
% 
%     %Estraggo la maschera etichettata
%     mask = localizer(img);    
% 
%     %Estrai le caratteristiche
%     [~,X_train_obj] = leaves_feature_extractor(mask, img, []);
%     X_train_leaves = [X_train_leaves; X_train_obj];
% end
% 
% % Azzero le righe dove sono presenti gli unknown per allenare il modello
% % Tolgo etichette e righe corrispettive
% 
% % Rimuovo righe unknown da dati di train
% X_train_no_unk = X_train_leaves;
% X_train_no_unk(GT_TRAIN_UNK_ROWS, :) = [];
% 
% % Rimuovo unknown dalla GT di train
% Y_train_no_unk = Gt_train_class;
% Y_train_no_unk(GT_TRAIN_UNK_ROWS, :) = [];
% 
% X_train_leaves = normalize_features(X_train_no_unk);


%% Estrazione dataset del test, senza UNKNOWN predictati precedentemente
disp("----Estrazioni dati per foglie testing-----");

% Trovo indici corrispondenti ad oggetti Unknown trovati
righe_unk = find(strcmp(predUnk_test, "Unknown"));

X_test_leaves = [];
for i = 1:N_IMMAGINI_TEST
    file_name = fullfile(TEST_PATH, ['test-', num2str(i), '.jpg']); 
    img = imread(file_name);
    disp(file_name);

    % Estraggo maschera 
    mask = localizer(img);   

    % Estraggo feature
    [predUnk_test, X_test_obj] = leaves_feature_extractor(mask, img, predUnk_test);
    X_test_leaves = [X_test_leaves; X_test_obj];
end

% Rimozione dati etichettati come UNK per creare la GT
%Y_test_no_unk = Gt_test_class;
%Y_test_no_unk(GT_TEST_UNK_ROWS, :) = [];

% la rimozione dei unk dai dati di testing viene fatta in leaves_feature_extractor
X_test_leaves = normalize_features(X_test_leaves);

%% Calcolo metriche
pred = predict(model, X_test_leaves);

% Metodo per unire predizione unknown e predizione foglie nell'ordine di
% etichettata
% (è normale che sia 0x1 string alla fine dell'iterazione del for)

pred_con_unkn = update_pred(pred, righe_unk);

% Calcolo delle metriche, con unknown inclusi
predictions_metrics(pred_con_unkn,Gt_test_class);


% Carico GT da file JSON
gt_path = "ground-truth\result.json";
coco_data = jsondecode(fileread(gt_path));
annotations = coco_data.annotations;

for i = 1:N_IMMAGINI_TEST
    file_name = fullfile(TEST_PATH, ['test-', num2str(i), '.jpg']); % Costruzione del percorso
    img = imread(file_name);
    disp(file_name);

    % Calcolo maschera
    mask_seg = localizer(img);  

    % Creo la maschera binaria dalle coordinate dell GT
    mask_gt = create_binary_mask(annotations, i);

    % Mostro BBOX con classe associata e coeff di Jaccard (IoU) globale
    [pred_con_unkn, pred_poly] = annotate_images(mask_seg, mask_gt, img, pred_con_unkn); 
end

%% Funzione normalizzazione
function X_normalized = normalize_features(X_train)
% Numero di colonne in X_train
[nRows, nCols] = size(X_train);

% Inizializza la matrice di output
X_normalized = zeros(nRows, nCols);

% Normalizza ogni colonna separatamente (per ogni feature)
for col = 1:nCols
    minVal = min(X_train(:, col)); 
    maxVal = max(X_train(:, col)); 

    % Normalizzazione 0 e 1 per la colonna
    X_normalized(:, col) = (X_train(:, col) - minVal) / (maxVal - minVal);
end
end

%% Funzione per unire predizione classi di foglie + unk
function risultato = update_pred(pred, righe_unk)
    righe_unk = sort(righe_unk);
    
    % Inizializza il vettore risultato
    risultato = strings(length(pred) + length(righe_unk), 1);
    posPred = 1;
    
    for i = 1:length(pred) + length(righe_unk)
        if ismember(i, righe_unk)
            risultato(i) = "Unknown";
        else
            risultato(i) = pred{posPred};
            posPred = posPred + 1;
        end
    end
end
