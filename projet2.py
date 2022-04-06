import pandas as pd

import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='whitegrid')
from PIL import Image
plt.ion()

st.title('Analyse des campagnes promotionnelles bancaires')
st.caption('par Karim SABER-CHERIF et Artem VALIULIN')
#image = Image.open(r"C:\Users\tyoma\Downloads\header.png")
#st.image(image, width=800)
st.markdown('Le jeu de données initial :')
 
df = pd.read_csv('/Users/karim/data/projet/bank.csv')
st.dataframe(df)

#####################################################################################
#####################################################################################

df_new = df.copy() # On enregistre df dans df_new pour ne pas corrompre les données de df
df_new.deposit = df_new.deposit.replace(['no', 'yes'], [0, 1])
df_new = df_new.drop(['duration'], axis=1)
data = df_new.drop('deposit', axis=1)
target = df_new['deposit']
df_new['campaign'] = df_new['campaign'].apply(lambda x : df_new.campaign.mean() if x > 35 else x)
# On remplace les variables binaires par 0 ou 1
data.housing = data.housing.replace(['no', 'yes'], [0, 1])
data.loan = data.loan.replace(['no', 'yes'], [0, 1])
data.default = data.default.replace(['no', 'yes'], [0, 1])
data = data.join(pd.get_dummies(data.marital, prefix='marital'))
data = data.join(pd.get_dummies(data.job, prefix='job'))
data = data.join(pd.get_dummies(data.contact, prefix='contact'))
data = data.join(pd.get_dummies(data.month, prefix='month'))
data = data.join(pd.get_dummies(data.poutcome, prefix='pout'))
data = data.join(pd.get_dummies(data.education, prefix='edu'))

data = data.drop(['marital', 'job', 'contact', 'month', 'poutcome', 'education'], axis=1)

# TEST DES MODELES
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# on split nos données
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

# On crée 6 instances à partir de 6 modèles de classifieurs différents
gbc = GradientBoostingClassifier()
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
ac = AdaBoostClassifier()
bc = BaggingClassifier()

# On entraine les modèles sur les données d'entrainement
gbc.fit(X_train, y_train)
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
ac.fit(X_train, y_train)
bc.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
# On prédit
y_pred_gbc = gbc.predict(X_test)
y_pred_dt = dt.predict(X_test)

# On affiche les score des modeles sur train et test, et fait une validation croisée à 5 échantillons
print('gradient boosting accuracy train score : ',metrics.accuracy_score(y_train, gbc.predict(X_train)).round(2))
print('gradient boosting accuracy test score : ',metrics.accuracy_score(y_test, y_pred_gbc).round(2))
print('decision tree accuracy train score : ',metrics.accuracy_score(y_train, dt.predict(X_train)).round(2))
print('decision tree accuracy test score : ',metrics.accuracy_score(y_test, y_pred_dt).round(2))
gbc_score = cross_val_score(gbc, X=X_train, y=y_train, cv=5, scoring='accuracy', n_jobs=-1)
dt_score = cross_val_score(dt, X=X_train, y=y_train, cv=5, scoring='accuracy', n_jobs=-1)
print('cross validation mean accuracy GBC: {0:.2f}%'.format(np.mean(gbc_score)*100))
print('cross validation mean accuracy DT: {0:.2f}%'.format(np.mean(dt_score)*100))
'cross validation mean accuracy GBC: {0:.2f}%'.format(np.mean(gbc_score)*100)

#affichage du score des modeles sur streamlit
#score = round(np.mean(dt_score)*100,2)
#st.caption('cross validation mean accuracy GBC:')
#st.caption(round(np.mean(dt_score)*100,2))

#####################################################################################
#####################################################################################

# Create a page dropdown 
page = st.sidebar.selectbox("Menu", ["Etude des variables", "Tests statistiques", "Machine learning"]) 

if page == "Etude des variables":

#poutcome
    st.markdown('Réponse des clients lors de la précédente campagne :')

    fig, ax = plt.subplots(figsize=(20,7))
    sns.countplot(df.poutcome, hue=df.deposit, palette=['#7B68EE', '#FF7F50'], alpha=0.75)
    plt.ylabel('Fréquence (%)')
    plt.title('poutcome vs deposit')
    _ = ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/df.shape[0]))
    st.pyplot(fig)

#job
    st.markdown('Fréquence des réponses selon le métier du client')

    fig, ax = plt.subplots(figsize=(20,9))
    sns.countplot(df.job, hue=df.deposit, palette=['#7B68EE', '#FF7F50'], alpha=0.75)
    plt.ylabel('Fréquence (%)')
    plt.title('job vs deposit')
    _ = ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/df.shape[0]))
    st.pyplot(fig)

#balance
    st.markdown('La réponse du client en fonction du montant de son compte bancaire :')

    fig, ax = plt.subplots(figsize=(20,7))
    ax = sns.countplot(pd.qcut(df.balance, [0, .25, .5, .75, .9, 1.]), hue=df.deposit, palette=['#7B68EE', '#FF7F50'], alpha=0.75)
    _ = ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/df.shape[0]))
    plt.ylabel('fréquence')
    plt.title('balance vs deposit')
    st.pyplot(fig)

elif page == "Tests statistiques":
    #pearson
    st.markdown('La matrice de corrélation (test de Pearson) :')

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True,cmap='viridis')
    plt.title('Coefficient de Pearson')
    st.pyplot(fig)


    from scipy.stats import chi2_contingency
    df_cat = df.select_dtypes('object')
    def V_Cramer(table, N):
        stat_chi2 = chi2_contingency(table)[0]
        k = table.shape[0]
        r = table.shape[1]
        phi_2 = max(0,(stat_chi2)/N - ((k - 1)*(r - 1)/(N - 1)))
        k_b = k - (np.square(k - 1) / (N - 1))
        r_b = r - (np.square(r - 1) / (N - 1))   
        return np.sqrt(phi_2 / min(k_b - 1, r_b - 1))

    dico = {}
    for col in df_cat.columns[df_cat.columns != 'deposit']:
        table = pd.crosstab(df_cat[col], df['deposit'])
        res = chi2_contingency(table)
        dico[col] = [res[0], res[1], res[2], V_Cramer(table, df.shape[0])]
    
    
    stats = pd.DataFrame.from_dict(dico).transpose()
    stats = stats.rename(columns={0:'chi 2', 1:'p-value', 2:'DoF', 3:'V de Cramer'})
    st.write(stats)
    st.write("Ce test nous donne des informations sur la corrélation entre les variables catégorielles et la variable cible.\
              16 Les variables ayant un V de Cramer compris entre 20 et 30 sont des variables très corrélées à la variable deposit, on a \
              entre autres: - housing, contact, month, poutcome.")
    
    st.write('Il est aussi possible de faire un test ANOVA entre les variables quantitatives et les\
variables qualitatives. ANOVA (ANalyse Of VAriance), compare l’écart des moyennes\
d’échantillons par rapport à la moyenne globale. Plus l’écart est important plus la\
variance est grande, et inversement. Le test renvoie la statistique F et la p-value, qui va\
nous servir à rejeter ou non l’hypothèse. En général, plus F est élevé, moins les\
variables sont corrélées, elles sont donc chacunes pertinentes.')
    import statsmodels.api
    result = statsmodels.formula.api.ols('balance ~ loan', data=df).fit()
    table = statsmodels.api.stats.anova_lm(result)
    st.write(table)
    st.write("Ici, on voit que la variable balance et la variable loan obtiennent une p-value très faible,\
on peut donc rejeter l’hypothèse H0 entre ces 2 variables.")




elif page == "Machine learning":

    fig, ax = plt.subplots(figsize=(20,6))
    sns.boxplot(df.previous)
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(20,6))
    sns.boxplot(df.campaign)
    st.pyplot(fig)

    st.write('On remarque certaines valeurs extrêmes. On peut s’y intéresser en affichant les lignes dans\
    le dataframe liées à ces valeurs.')
    st.write('Quand previous > 35')
    st.write(df_new[df_new['previous']>35])
    st.write('Quand campaign > 35')
    st.write(df_new[df_new['campaign']>35])


    st.write('Il vient ensuite le feature engeneering des variables :')

    st.write('On remplace les variables binaires par 0 ou 1')
    st.write('On crée des variables indicatrices à partir des variables catégorielles')
    st.write('On supprimes les variables dont on a plus besoin')
    st.write(data.head())

    model = st.selectbox(label="Choix du modèle", options=["Gradient Boosting", "Decision Tree",
     'Random Forest Classifier', 'Logistic Regression'])

    def get_model(model):
        if model == "Gradient Boosting":
            model = GradientBoostingClassifier(n_estimators = 200,
                                  learning_rate=0.1,
                                  max_depth = 6,
                                  random_state = 234)
        elif model == "Decision Tree":
            model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=234)
        elif model == 'Random Forest Classifier':
            model = RandomForestClassifier()
        elif model == 'Logistic Regression':
            model = LogisticRegression()

        model.fit(X_train, y_train)
        score = model.score(X_test,y_test)

        return score

    st.write("Score test :", get_model(model))

    # cross validation

    gbc_score = cross_val_score(gbc, X=X_train, y=y_train, cv=5, scoring='accuracy', n_jobs=-1)
    dt_score = cross_val_score(dt, X=X_train, y=y_train, cv=5, scoring='accuracy', n_jobs=-1) 
    st.write('Cross validation score GradientBoosting :', gbc_score.mean(), '%')  
    st.write('Cross validation score DecisionTree :', dt_score.mean(), '%') 


    # feature importance
    st.markdown("L'importance des variables déterminées par les modèles :")

    feature = {}
    for feat, imp in zip(data.columns, dt.feature_importances_):
        feature[feat] = imp
    
    tab = pd.DataFrame.from_dict(feature, orient='index', columns=['importance'])
    tab = tab.sort_values(by='importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(0, 10)
    sns.barplot(tab['importance'].head(10), x, orient='h')
    plt.yticks(x, tab.index[0:10])
    plt.title('Feature importances (DecisionTree)')
    st.pyplot(fig)

    feature = {}
    for feat, imp in zip(data.columns, gbc.feature_importances_):
        feature[feat] = imp
    
    tab = pd.DataFrame.from_dict(feature, orient='index', columns=['importance'])
    tab = tab.sort_values(by='importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(0, 10)
    sns.barplot(tab['importance'].head(10), x, orient='h')
    plt.yticks(x, tab.index[0:10])
    plt.title('Feature importances (Gradient Boosting)')
    st.pyplot(fig)

    st.write("Les 3 variables les plus importantes en termes d’informations pertinentes est poutcome=\
success, contact=unknown et age pour le modèle d’arbre de décision.\
Pour le modèle de gradient boosting, on a : balance, poutcome (success) et age.\
Les variables age et poutcome sont donc des informations à privilégier lors d’une\
campagne, notamment le fait que les clients contactés qui ont répondu positivement à une\
précédente campagne, ont plus de chance de répondre positivement à une autre\
campagne. Ils seront donc à contacter en priorité pour de futures campagnes. On peut\
aussi s’en assurer visuellement :")


    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(df['poutcome'][df['poutcome']=='success'], hue=df.deposit)
    st.pyplot(fig)

    st.write('En effet, les clients ayant déjà profiter d’une offre antérieure, ont répondu en grande\
majorité positivement à la campagne étudiée.')


    from sklearn import tree
    dt = DecisionTreeClassifier(max_depth=6)
    dt.fit(X_train, y_train)
    st.write(dt.score(X_test, y_test))
    fig, ax = plt.subplots(figsize=(20,8))
    tree.plot_tree(dt, max_depth=2, filled=True, feature_names=X_train.columns, class_names=True)
    st.pyplot(fig)


    y_pred = gbc.predict(X_test)
    st.code(metrics.classification_report(y_test, y_pred))
    st.code(pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))

    st.write('Traçons maintenant la courbe ROC. La courbe ROC affiche la “sensibilité” (Vrai positif) en\
fonction de “antispécificité” (Faux positif). On calcul en fait l’air sous la courbe (AUC, area\
under the curve). Si cette valeur est de 0,5 (50%), le modèle est aléatoire, plus l’aire est\
importante, plus notre modèle sera performant et arrivera à classifier correctement.')


    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    fig, ax = plt.subplots(figsize=(10,8))

    y_pred = dt.predict(X_test)
    probs = dt.predict_proba(X_test)
    fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label=1)
    plt.plot(fpr, tpr, color='red', lw=3, label='Modèle DT (auc = {}%)'.format(auc(fpr, tpr).round(2)*100))

    y_pred = gbc.predict(X_test)
    probs = gbc.predict_proba(X_test)
    fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label=1)
    plt.plot(fpr, tpr, color='orange', lw=3, label='Modèle GB (auc = {}%)'.format(auc(fpr, tpr).round(2)*100))

    x = np.linspace(0,1,50)
    plt.plot(x, x, 'b--', lw=3, label='Aléatoire (auc = 0.5)')
    plt.xlim(0,1.02)
    plt.ylim(0,1.02)
    plt.legend(loc='lower right')
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrais positifs')
    plt.title('Courbe ROC')
    plt.grid(False)
    st.pyplot(fig)


