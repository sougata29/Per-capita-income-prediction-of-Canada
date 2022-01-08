import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("canada_per_capita_income.csv")
plt.xlabel("year")
plt.ylabel("per capita income")
cap = df.per_capita_income_USDoller
reg = linear_model.LinearRegression()
new_df = df.drop('per_capita_income_USDoller',axis='columns')
reg.fit(new_df,cap)
plt.scatter(df.year,cap)
plt.plot(df.year,reg.predict(df[["year"]]))
plt.show()
#prediction of per capita income in the year 2020
predict = reg.predict([[2020]])
print(predict)
