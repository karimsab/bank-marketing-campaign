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

st.title("Phase de l'analyse")


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

#pearson
st.markdown('La matrice de corrélation (test de Pearson) :')

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), annot=True,cmap='viridis')
plt.title('Coefficient de Pearson')
st.pyplot(fig)


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

# TEST DES MODELE SANS STANDARDISER LES VARIABLES NUMERIQUES
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
score = round(np.mean(dt_score)*100,2)
st.caption('cross validation mean accuracy GBC:')
st.caption(round(np.mean(dt_score)*100,2))
# feature importance
st.markdown("L'importance des variables déterminées par les modèles :")
