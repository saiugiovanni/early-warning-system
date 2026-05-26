# External file with some useful functions for the project
# Import libraries needed for the functions to work 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shap

# This function compute the best threshold according to the AUC score
def OptimalThreshold_AUC(pdf, true_labels):
    # initialization
    sensitivity = np.zeros(len(pdf))
    specificity = np.zeros(len(pdf))
    i=0
    # choose threshold between the pdf values
    for threshold in pdf:
        Pred_Labels = 1* (pdf < threshold)
        # evaluate prediction: true/false positive, true/false negative
        tp = np.sum((Pred_Labels == 1) & (true_labels==1))
        tn = np.sum((Pred_Labels == 0) & (true_labels==0))
        fp = np.sum((Pred_Labels == 1) & (true_labels==0))
        fn = np.sum((Pred_Labels == 0) & (true_labels==1))
        # Sensitivity, Specificity
        sensitivity[i] = tp / (tp + fn) #like recall
        specificity[i] = tn / (fp + tn)
        i=i+1
    # plot roc curve
    sorted = np.array([1-specificity, sensitivity]).T
    sorted = sorted[np.argsort(sorted[:,0])]
    AUC = np.trapz(sorted[:,1], sorted[:,0]) 
    print(AUC)
    plt.plot(sorted[:,0], sorted[:,1],'b-', label = 'ROC Curve (AUC = '+str(AUC)+' )')
    x = np.linspace(0,1, num = len(specificity), )
    plt.plot(x,x,'k', label = 'Trivial')
    # select threshold optimizing ROC curve
    distance = np.linalg.norm( np.array((1-specificity, sensitivity)).T - np.array([0, 1]).T)
    index = np.where(distance == np.min(distance))
    best_eps = pdf[index]
    plt.plot(1-specificity[index], sensitivity[index],'ro', label = 'Operating point')
    plt.legend(loc="lower right")
    return best_eps, specificity, sensitivity

# This is a function to choose the threshold optimizing the F1 beta score
def OptimalThreshold_fbeta(ycval, p, beta):
    # Find the best threshold (epsilon) to use for selecting anomalies
    # ycval = responses Y=1 if anomaly, 0 otherwise
    # p = density
    # initialization
    bestEpsilon = 0
    bestF1 = 0
    # choose threshold in p
    for epsilon in p:
        predictions = 1*(p < epsilon)
        # compute true positives, false negatives and false positives
        # according to predictions
        tp = np.sum((predictions == 1) & (ycval == 1))
        fp = np.sum((predictions == 1) & (ycval == 0))
        fn = np.sum((predictions == 0) & (ycval == 1))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        # This is the formula for the f1 beta score
        # The higher beta the more weight is given to recall
        F1 = (1+beta**2) * prec * rec / (prec*(beta**2) + rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1

# This is a function to plot important scores and the confusion matrix   
def PlotScores(Pred_Labels, true_labels):
    # Given predictions and true labels compute the various indicators
    tp = sum((Pred_Labels == 1) & (true_labels==1))
    tn = sum((Pred_Labels == 0) & (true_labels==0))
    fp = sum((Pred_Labels == 1) & (true_labels==0))
    fn = sum((Pred_Labels == 0) & (true_labels==1))
    acc = (tp + tn)/(tp+fp+tn+fn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    F1 = 2 * prec * rec / (prec + rec)
    # Print them out
    print('---------- Scores ---------')
    print("\nAccuracy = %.2f %% \n" % (acc))
    print("Precision = %.2f\n" %prec)
    print("Recall = %.2f\n" % rec)
    print("F1 score = %.2f\n" % F1)
    # Visualize the confusion matrix as a heatmap
    conf_mat = confusion_matrix(true_labels, Pred_Labels)
    ax = sns.heatmap(conf_mat, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    # Display the visualization of the Confusion Matrix.
    plt.show()

# This is a function to plot the force plot of the library
# shap on one single observation (obs_ind), selecting only a number
# (n_features) of features that contribute positively and negatively.
# For example, if n_features = 3, in the plot we will see three
# positive contributions and three negative contributions.
def shap_plot_single_obs_tree(obs_ind, n_features, model_predictions, classifier, xtest):
    # obs_ind: number of observation
    # n_features: number of features to plot
    # model_predictions: the predictions of the model, i.e. 0 or 1
    # classifier: the classifier passed must be a random forest
    # xtest: the test values from which obs_ind is chosen. Should be
    #        a pandas dataframe.

    # Check if obs_ind has been predicted as 1 or 0
    if model_predictions[obs_ind] == 1:
        flag=1
    if model_predictions[obs_ind] == 0:
        flag=0

    # Create the explainer object of a random forest predictor
    rf_explainer = shap.TreeExplainer(classifier, xtest)
    
    # Store feature names
    u_cols = xtest.columns

    # Get the shapley values for the whole test set
    shap_val = rf_explainer.shap_values(xtest, check_additivity=False)

    # Get the shapley value of interest, according to the model prediction
    # and to the observation of interest
    shap_user = shap_val[flag][obs_ind,:]
    shap_user_importance = np.argsort(shap_user) # from low to high
    top_user_n = n_features # store the most important features

    # Store the negative contributing features. Remember: shap_user_importance
    # is sorted from low to high
    neg_cols = [ u_cols[shap_user_importance[c]] for c in range(top_user_n)]
    neg_vals = [ shap_user[shap_user_importance[c]] for c in range(top_user_n)]
    neg_vals = np.array(neg_vals)
    neg_indexes = [ shap_user_importance[c] for c in range(top_user_n)]

    # Store the positive contributing features. Remember: shap_user_importance
    # is sorted from low to high
    pos_cols = [ u_cols[shap_user_importance[-(c+1)]] for c in range(top_user_n)]
    pos_vals = [ shap_user[shap_user_importance[-(c+1)]] for c in range(top_user_n)]
    pos_vals=np.array(pos_vals)
    pos_indexes = [ shap_user_importance[-(c+1)] for c in range(top_user_n)]

    # Keep in main_feat the negative and the positive features of interest
    main_feat = neg_indexes[:]
    main_feat.extend(pos_indexes[:])
    main_col_names = [ u_cols[_] for _ in main_feat]
    # Print out results
    print("Features contributing positively:")
    print(pos_cols)
    print("Features contributing negatitively:")
    print(neg_cols)
    #shap.initjs() This command is needed only if you don't want
    # to plot with matplotlib
    shap.force_plot(rf_explainer.expected_value[flag],
                     shap_val[flag][obs_ind,main_feat], 
                     xtest.iloc[obs_ind,main_feat], 
                     feature_names=main_col_names,
                     text_rotation=90, 
                     figsize=(24,3), 
                     matplotlib=True)
   
    return plt.show()
