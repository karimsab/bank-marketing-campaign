import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


st.title('Analyse des campagnes promotionnelles bancaires')
st.caption('par Karim SABER-CHERIF et Artem VALIULIN')

df = pd.read_csv("https://raw.githubusercontent.com/karimsab/bank-marketing-campaign/main/bank.csv", sep=",", header=0)

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

#####################################################################################
#####################################################################################

page = st.sidebar.selectbox("Menu", ["Etude des variables", "Tests statistiques", "Machine learning"]) 

###### affichage des différentes pages ######
###### page études des variables ######

if page == "Etude des variables":   

    st.image('https://user-images.githubusercontent.com/62601686/165300401-3f1257db-3a61-419a-bf7a-840c3c013ff7.png', width=800)
    st.text("Pour ce jeu de données, nous avons des données personnelles sur des clients \n\
d’une banque qui ont été “télémarketés” pour souscrire à un produit que \n\
l’on appelle un 'dépôt à terme'. Lorsqu’un client souscrit à ce produit, il place \n\
une quantité d’argent dans un compte spécifique et ne pourra pas toucher ces fonds \n\
avant l’expiration du terme. En échange, le client reçoit des intérêts de la \n\
part de la banque à la fin du terme.")
    st.markdown('Le jeu de données initial :')
    st.dataframe(df)
    st.markdown('Explication des variables :')
    st.text("1 - age : âge du client \n\
2 - job : métier du client \n\
3 - marital : statut marital du client \n\
4 - education : niveau d'étude du client \n\
5 - default : si le client à un crédit impayé \n\
6 - balance : somme d'argent sur le compte bancaire \n\
7 - housing : si le client a un emprûnt immobilier \n\
8 - loan : si le client à un crédit en cours \n\
9 - contact : type de contact (cellular, phone, unknown) \n\
10 - day : le jour du mois où le client a été contacté \n\
11 - month : le mois où le client a été contacté \n\
12 - duration : le temps en ligne avec le client \n\
13 - campaign : combien de fois le client a été contacté \n\
                au cours de cette campagne \n\
14 - pdays : combien de jours se sont écoulés depuis le \n\
             dernier contact \n\
15 - previous : combien de fois le client a été contacté \n\
                au cours de la précedente campagne \n\
16 - poutcome : résultat de la précédente campagne \n\
17 - deposit :  résultat de la présente campagne")

    # deposit
    labels = ['No', 'Yes']
    values = df.deposit.value_counts(normalize=True)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial',
                             pull=[0, 0.1])])
    fig.update_layout(title_text="Distribution de la variable cible 'deposit'",
                 margin=dict(t=80, b=10, l=10, r=10))
    fig.update_traces(marker=dict(colors=['lightcoral', 'seagreen'], line=dict(color='#000000', width=1)))

    st.plotly_chart(fig)

    st.write("48% des sondés ont répondus à l'offre positivement")

    # age
    fig = go.Figure()

    fig.add_trace(go.Histogram(x = df['age'][df['deposit'] == 'no'],
                          name = 'no',  
                          marker_color = 'lightcoral'))
    
    fig.add_trace(go.Histogram(x = df['age'][df['deposit'] == 'yes'],
                          name = 'yes',  
                          marker_color = 'seagreen'))

    fig.update_layout(title = 'Distribution de la variable age',
                 xaxis_title = 'age',
                  yaxis_title = 'Count',
                 autosize = False,
                 width = 800, 
                 height = 600,
                 template='simple_white')

    st.plotly_chart(fig)

    st.write("Entre 30 et 60 ans, les clients déclinent l'offre plus souvent")

    # balance
    fig = go.Figure()
    dfq = df.copy()
    dfq['balance'] = pd.qcut(dfq.balance, [0, .25, .5, .75, .9, 1.]).astype('str')
    fig.add_trace(go.Histogram(x = dfq['balance'][dfq['deposit'] == 'yes'],
                          name = 'yes', # Label 
                          marker_color = 'seagreen',
                          histnorm='percent',
                          opacity=0.8)) # Couleur des barres 

    fig.add_trace(go.Histogram(x = dfq['balance'][dfq['deposit'] == 'no'],
                          name = 'no', # Label 
                          marker_color = 'lightcoral',
                          histnorm='percent',
                          opacity=0.8)) # Couleur des barres 

    fig.update_layout(title = 'Distribution de la variable balance',
                 xaxis_title = 'balance',
                 yaxis_title = 'frequency',
                 width = 800, 
                 height = 600,
                 template='simple_white')

    st.plotly_chart(fig)

    st.write("Sans surprise, les clients ayant peu d'argent sur leur compte répondent \
en moyenne négativement à l'offre. Il serait donc préférable de contacter les clients \
avec une somme minimal sur leur compte bancaire.")

    # day
    fig = go.Figure()

    fig.add_trace(go.Histogram(x = df['day'][df['deposit'] == 'yes'],
                          name = 'yes',  
                          marker_color = 'seagreen'))

    fig.add_trace(go.Histogram(x = df['day'][df['deposit'] == 'no'],
                          name = 'no',  
                          marker_color = 'lightcoral'))

    fig.update_layout(title = 'Distribution de la variable day',
                 xaxis_title = 'day',
                  yaxis_title = 'Count',
                 autosize = False,
                 width = 800, 
                 height = 600,
                 template='simple_white')

    st.plotly_chart(fig)    

    st.write("La variable 'day' représente le jour du mois où le client a été appelé. On observe qu'en moyenne \
si les clients ont été appelé dans les deux dernières semaines du mois, ils seront plus suceptible \
de répondre négativement à l'offre.")

    # month
    fig = go.Figure()

    fig.add_trace(go.Histogram(x = df['month'][df['deposit'] == 'yes'],
                            name = 'yes',  
                            marker_color = 'seagreen'))

    fig.add_trace(go.Histogram(x = df['month'][df['deposit'] == 'no'],
                            name = 'no',  
                            marker_color = 'lightcoral'))

    fig.update_layout(title = 'Distribution de la variable month',
                    xaxis_title = 'month',
                    yaxis_title = 'Count',
                    autosize = False,
                    width = 800, 
                    height = 600,
                    template='simple_white')

    st.plotly_chart(fig)

    st.write("On voit nettement que certains moins, les clients seront moins réceptifs \
à l'offre promotionnelle. En mai, seule 1 personne sur 3 à effectuer un dépôt à terme.")

    # education
    fig = go.Figure()

    fig.add_trace(go.Histogram(x = df['education'][df['deposit'] == 'yes'],
                            name = 'yes',  
                            marker_color = 'seagreen'))

    fig.add_trace(go.Histogram(x = df['education'][df['deposit'] == 'no'],
                            name = 'no',  
                            marker_color = 'lightcoral'))

    fig.update_layout(title = 'Distribution de la variable education',
                    xaxis_title = 'education',
                    yaxis_title = 'Count',
                    autosize = False,
                    width = 800, 
                    height = 600,
                    template='simple_white')

    st.plotly_chart(fig)

    st.write("Ici, on peut dire que les personnes ayant fait des études supérieurs ont plus tendance \
à accepter l'offre que s'ils s'étaient arrêtés au secondaire ou en primaire.")

    # marital
    fig = go.Figure()

    fig.add_trace(go.Histogram(x = df['marital'][df['deposit'] == 'yes'],
                            name = 'yes',  
                            marker_color = 'seagreen'))

    fig.add_trace(go.Histogram(x = df['marital'][df['deposit'] == 'no'],
                            name = 'no',  
                            marker_color = 'lightcoral'))

    fig.update_layout(title = 'Distribution de la variable marital',
                    xaxis_title = 'marital',
                    yaxis_title = 'Count',
                    autosize = False,
                    width = 800, 
                    height = 600,
                    template='simple_white')

    st.plotly_chart(fig)

    # job
    fig = go.Figure()

    fig.add_trace(go.Histogram(x = df['job'][df['deposit'] == 'yes'],
                            name = 'yes',  
                            marker_color = 'seagreen'))

    fig.add_trace(go.Histogram(x = df['job'][df['deposit'] == 'no'],
                            name = 'no',  
                            marker_color = 'lightcoral'))

    fig.update_layout(title = 'Distribution de la variable job',
                    xaxis_title = 'job',
                    yaxis_title = 'Count',
                    autosize = False,
                    width = 800, 
                    height = 600,
                    template='simple_white')

    st.plotly_chart(fig)

    st.write("Certains métiers affichent des tendances nettes : les cols bleus ne sont pas \
        à prioriser lors de contacts par la banque car peu intêressés par l'offre. Au contraire \
        les retraités ou les étudiants sont des clients prioritaires.")

    # poutcome
    fig = px.parallel_categories(df, dimensions=['poutcome', 'deposit'],
                             color_continuous_scale=px.colors.sequential.Inferno)

    st.plotly_chart(fig)

    st.write("La variable 'poutcome' a 4 modalités, dont une se nomme 'unknown', soit une valeur \
     inconnu, cependant cette variable nous donne des informations importantes sur les résultats \
         de l'enquête. Essayons d'en savoir plus")
    
    st.code("df.poutcome[df.previous == 0].unique() --> unknown")
        
    st.write("En effet, la variable 'previous' nous indique si la personne a déjà été contacté avant \
         cette campagne et si oui, combien de fois. \
             Après filtrage du dataframe pour les personne n'ayant jamais été contacté, la modalité 'unknown' est affichée, ce qui indique \
             que ce sont des personnes qui sont contacté pour la 1ère fois")

###### page tests statistiques ######

elif page == "Tests statistiques":
    #pearson
    st.markdown('**La matrice de corrélation (test de Pearson) :**')

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True,cmap='viridis')
    plt.title('Coefficient de Pearson')
    st.pyplot(fig)


    st.write('Le tableau affiché nous donne les informations suivantes : statistique du test, p-value, \
degré de liberté, V de Cramer (coefficient de corrélation du 𝜒2)')

    st.latex(r'''test  du  \chi 2''')
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
    st.write("Ce test nous donne des informations sur la corrélation entre les variables catégorielles et la variable cible. \
16 Les variables ayant un V de Cramer compris entre 20 et 30 sont des variables très corrélées à la variable \
deposit, on a entre autres: - housing, contact, month, poutcome.")

    st.markdown('**test ANOVA**')
    st.write('Il est aussi possible de faire un test ANOVA entre les variables quantitatives et les \
variables qualitatives. ANOVA (ANalyse Of VAriance), compare l’écart des moyennes \
d’échantillons par rapport à la moyenne globale. Plus l’écart est important plus la \
variance est grande, et inversement. Le test renvoie la statistique F et la p-value, qui va \
nous servir à rejeter ou non l’hypothèse. En général, plus F est élevé, moins les \
variables sont corrélées, elles sont donc chacunes pertinentes.')
    import statsmodels.api
    result = statsmodels.formula.api.ols('balance ~ loan', data=df).fit()
    table = statsmodels.api.stats.anova_lm(result)
    st.write(table)
    st.write("Ici, on voit que la variable balance et la variable loan obtiennent une p-value très faible, \
on peut donc rejeter l’hypothèse H0 entre ces 2 variables.")

###### page machine learning ######

elif page == "Machine learning":
    st.write('Essayons dans cette partie, de transformer les données afin que le modèle les exploitent \
au mieux. Concentrons nous sur la répartition des 2 variables quantitatives – previous et \
campaign – suivantes, affichées à l’aide d’un boxplot :')
    st.write('boxplot "previous"')

    fig, ax = plt.subplots(figsize=(20,6))
    sns.boxplot(df.previous)
    st.pyplot(fig)
    st.write('boxplot "campaign"')
    fig, ax = plt.subplots(figsize=(20,6))
    sns.boxplot(df.campaign)
    st.pyplot(fig)

    st.write('On remarque certaines valeurs extrêmes. On peut s’y intéresser en affichant les lignes dans \
    le dataframe liées à ces valeurs.')
    st.write('Quand previous > 35')
    st.write(df[df['previous']>35])
    st.write('Pour rappel, previous nous indique le nombre de fois que le client a été appelé, avant cette \
campagne. 2 clients sur 5 ont répondu positivement à l’offre.')
    st.write('Quand campaign > 35')
    st.write(df[df['campaign']>35])

    st.write("Pour ces clients, aucun n'a répondu positivement à la campagne. \
Ces valeurs ne représentent pas la majorité des clients, il serait utile de majorer la variable \
campaign, pour le faire, on remplacera les valeurs extrêmes par la moyenne de la variable. \
On garde les valeurs extrêmes de previous car certains ont répondus positivement, c’est \
donc des informations utiles pour savoir quel type de client est susceptible de faire un \
dépôt après la campagne. Par la même occasion, on va créer une copie de df pour nos \
prochaines transformations, et aussi, binarisé la variable deposit.")


    st.markdown('**Il vient ensuite le feature engeneering des variables :**')

    st.write('On remplace les variables binaires par 0 ou 1')
    st.write('On crée des variables indicatrices à partir des variables catégorielles')
    st.write('On supprimes les variables dont on a plus besoin')
    st.write(data.head())

    st.write("On peut dorénavant appliquer nos modèles de classification sur nos données. Avant cela, \
on va diviser notre jeu de données en un set d'entraînement et un set de test, pour pouvoir \
ensuite vérifier la précision de nos modèles via plusieurs métriques. \
Essayons plusieurs modèles pour voir les plus performants.")

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
    st.markdown('**Validation croisée**')
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
success, contact=unknown et age pour le modèle d’arbre de décision. \
Pour le modèle de gradient boosting, on a : balance, poutcome (success) et age. \
Les variables age et poutcome sont donc des informations à privilégier lors d’une \
campagne, notamment le fait que les clients contactés qui ont répondu positivement à une \
précédente campagne, ont plus de chance de répondre positivement à une autre \
campagne. Ils seront donc à contacter en priorité pour de futures campagnes. On peut \
aussi s’en assurer visuellement :")


    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(df['poutcome'][df['poutcome']=='success'], hue=df.deposit)
    st.pyplot(fig)

    st.write('En effet, les clients ayant déjà profiter d’une offre antérieure, ont répondu en grande \
majorité positivement à la campagne étudiée.')

    st.markdown('**Arbre de décision :**')
    from sklearn import tree
    dt = DecisionTreeClassifier(max_depth=6)
    dt.fit(X_train, y_train)
    st.write('score decision tree')
    st.write(dt.score(X_test, y_test))
    fig, ax = plt.subplots(figsize=(20,8))
    tree.plot_tree(dt, max_depth=2, filled=True, feature_names=X_train.columns, class_names=True)
    st.pyplot(fig)
    
    y_pred = gbc.predict(X_test)
    st.write("On y voit les nœuds de décision, qui se splitent pour effectuer la classification selon un \n\
certain seuil par variable. Lorsque le nœud ne se split plus, on parle alors de leaf (feuille). On peut \n\
modifier ces paramètres afin de contrôler pour en optimiser son efficacité de classification : par exemple, \n\
un nombre de noeuds trop important amènerait à du surapprentissage.")
    st.write("On peut afficher une matrice de confusion pour s'aider dans la compréhension du modèle:")
    st.image("https://user-images.githubusercontent.com/62601686/165782931-eb29223f-6570-4b84-850e-477fae038118.png", width=600)
    st.write("Plusieurs critères de performances découlent de la matrice de confusion, ainsi on a :\n\
1 - **Le Rappel** (recall) : mesure le taux de vrais positifs, défini par : ")
    st.latex(r'''Rappel = \frac{TP}{TP + FN}''')
    st.write("C'est la capacité de notre modèle à bien idientifier les clients ayant souscris à l'offre ainsi que les \n\
faux négatifs, soit les personnes qu'on pense avoir fait un dépôt à terme mais qui n'ont pas souscris à l'offre.")
    st.write("2 - **La Précision** : mesure la performance du modèle à détecter les clients qui vont faire \n\
un dépôt à terme. On a des informations sur les faux positifs aussi, c'est-à-dire ceux qui vont faire un \n\
dépôt d'argent mais qui ne seront pas détecter. Donc cela permet de limiter les actions envers ces clients \n\
car ils ne font pas partie des personnes sceptiques à l'offre et donc sur qui il faudrait déployer d'autres \n\
ressources.")
    st.latex(r'''Precision = \frac{TP}{TP + FP}''')
    st.code(pd.crosstab(y_test, y_pred, rownames=['Classe prédite'], colnames=['Classe réelle']))
    st.write("Le rapport de classification condense ces informations, on a en plus le F-1 score, une \n\
métrique qui combine le rappel et la précision :")
    st.code(metrics.classification_report(y_test, y_pred))
    prec = round(metrics.precision_score(y_test, y_pred),2)*100
    rec = round(metrics.recall_score(y_test, y_pred),2)*100
    

    st.markdown('**Courbe ROC :**')
    st.write('Traçons maintenant la courbe ROC. La courbe ROC affiche la “sensibilité” (Vrai positif) en \n\
fonction de “antispécificité” (Faux positif). On calcul en fait l’air sous la courbe (AUC, area under the curve). \n\
Si cette valeur est de 0,5 (50%), le modèle est aléatoire, plus l’aire est importante, plus notre modèle sera \n\
performant et arrivera à classifier correctement.')


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

    st.markdown('**GridSearchCV :**')


    from sklearn.model_selection import GridSearchCV

    with st.echo():
        hyperparameter = dict(max_depth = [5, 15], min_samples_leaf = [3, 10])

        gridP = GridSearchCV(gbc, hyperparameter, cv = 3, verbose = 1, n_jobs = -1)
        grille = gridP.fit(X_train, y_train)

    st.write('meilleurs paramètres selectionnés :')
    st.code(gridP.best_params_)
    y_pred = gridP.predict(X_test)
    st.write('gradient boosting accuracy test score with best param : ')
    st.code(metrics.accuracy_score(y_test, y_pred).round(2))
