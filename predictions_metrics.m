function [mean_precision, mean_recall] = predictions_metrics(pred, gt)
    % Calcola la matrice di confusione
    figure(),
    cm = confusionchart(gt, pred);
    
    confMat = cm.NormalizedValues; 
    class = unique([pred, gt]);
   
    % Estrai TP, FP, FN, TN 
    TP = diag(confMat);                
    FP = sum(confMat, 1)' - TP;       
    FN = sum(confMat, 2) - TP;        
    TN = sum(confMat, 'all') - sum(TP) - sum(FP) - sum(FN);  
    
    % Calcolo dell'accuracy
    accuracy = (sum(TP) + TN) / sum(confMat, 'all');

    % Precision e Recall per ogni classe
    precision = TP ./ (TP + FP);
    recall = TP ./ (TP + FN);
    
    disp(cm);
    
    % Mostra metriche
    disp('Precision per classe:');
    disp(precision);
    disp('Recall per classe:');
    disp(recall);
    disp('Accuracy:');
    disp(accuracy);

    % Precisione media e Recall medio
    mean_precision = mean(precision);
    mean_recall = mean(recall);

    % Crea tabelle media recall e media precisione per ogni classe
    T = table(class, precision, recall, 'VariableNames', {'Classe', 'Precisione', 'Recall'});
    disp('Tabella delle metriche per classe:');
    disp(T);

    % In base alla versione di matlab, la stampa della tabella può dare
    % problemi
    % uitable('Data',T{:,:},'ColumnName',T.Properties.VariableNames,...
    % 'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
end