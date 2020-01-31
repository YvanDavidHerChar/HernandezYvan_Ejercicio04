import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 
import itertools as it

#Importando los datos
data = pd.read_csv('Cars93.csv')

#Organizando las cosas
Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
X = np.array(data[columns])

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_scaled, Y, test_size=0.3)



var = len(columns)
indexs= np.argsort(columns)
epa = 0
eso = 0
coeficientes = np.zeros(var+1)
for j in range(var):
    perm = np.array(list(it.combinations(indexs,j+1)))
    for i in range(len(perm)):
        new_X = X_train[:,perm[i]]
        regresion = sklearn.linear_model.LinearRegression()
        regresion.fit(new_X, Y_train)
        new_X_test = X_test[:,perm[i]]
        a = regresion.score(new_X_test,Y_test)
        if a>epa:
            epa=a
            coeficientes = regresion.coef_
        plt.scatter(j+1,a)
        
ii = np.argsort(np.abs(coeficientes))
for i in ii:
    print(columns[i], coeficientes[i])

plt.xlabel("X")
plt.ylabel("R cuadrado")
plt.title("Encontrando el Numero de Variables")
plt.savefig("nparams.png", bbox_inches='tight')

