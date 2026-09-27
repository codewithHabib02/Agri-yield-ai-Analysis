import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
df=pd.read_csv('Production_Crops_Livestock_E_All_Data.csv',low_memory=False)

print("\n" + "="* 70)
print("Head Of The Dataset:" )
print(df.head())
print("\n" + "="* 70)

print("\n" + "="* 70)
print("Infor About The Dataset: ")
print(df.info())
print("\n" + "="* 70)

print("\n" + "="* 70)
print("Columns Of The Dataset: ")
print(df.columns.tolist())
print("\n" + "="* 70)

print("\n" + "="* 70)
print("Dtypes Of The Dataset: ")
print(df.dtypes)
print("\n" + "="* 70)

print("\n" + "="* 70)
print("Missing Value Of The Dataset: ")
print(df.isnull().sum())
print("\n" + "="* 70)

print("\n" + "="* 70)
print("Total Missing Value Of The Dataset: ")
print(df.isnull().sum().sum())
print("\n" + "="* 70)

pd.set_option("display.max_rows",None)
print("\n" + "="* 70)
print("nPercentage Of Missing Value Of The Dataset: ")
print(df.isnull().sum()/len(df)*100)
print("\n" + "="* 70)

print("\n" + "="* 70)
print("Duplicates Of The Dataset: ")
print(df.duplicated().sum())
print("\n" + "="* 70)

print("\n" + "="* 70)
print("Description Of The Dataset: ")
print(df.describe())
print("\n" + "="* 70)

cat_cols=df.select_dtypes(include=['category','object']).columns.tolist()
for c in cat_cols:
    print("\n" + "="*70)
    print("Categorical Analysis")
    print(c,df[c].value_counts(dropna=False))
    print(c,df[c].nunique(),df[c].unique()[:10])

numeric_cols=df.select_dtypes(include=['int','float']).columns.tolist()
for k in numeric_cols:
    print("\n" + "="*70)
    print("Numeric Analysis")
    Q1=df[k].quantile(0.25)
    Q3=df[k].quantile(0.75)
    QIR=Q3-Q1
    outliers=((df[k]<Q1-1.5*QIR) | (df[k]>Q3 + 1.5 *QIR))
    print(f"{k}:{outliers.sum()} outlier")    

corr=df[numeric_cols].corr()
sns.heatmap(corr,annot=True,fmt='.2f')    
plt.close()

cat_cls=df.select_dtypes(include=['category','object']).columns.tolist()
for c in cat_cls:
    top_cols=df[c].value_counts(normalize=True).head(10)*100
    plt.figure(figsize=(10,5))
    top_cols.plot(kind='bar')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close()


col_to_drop=[c for c in df.columns if c.endswith(('F','N')) or c in 
             ['Area Code','Area Code (M49)','Item Code','Item Code (CPC)','Element Code']]
df=df.drop(columns=col_to_drop)

print("\n" + "="* 70)
print("Total Missing Value Of The Dataset: ")
print(df.isnull().sum().sum())
print("\n" + "="* 70)

year_cols=[c for c in df.columns if c.startswith('Y')]
df=df.melt(
    id_vars=['Area','Item','Element','Unit'],
    value_vars=year_cols,
    var_name='Year',
    value_name='Value'
)

df['Year']=df['Year'].str.replace('Y','').astype(int)

df=df.pivot_table(
    index=['Area','Item','Year'],
    columns='Element',
    values='Value',
    aggfunc='mean'
).reset_index()

df=df.dropna(subset=['Yield'])
df=df.drop(columns=['Stocks','Yield/Carcass Weight'])


x=df.drop(columns=['Yield','Laying','Milk Animals','Producing Animals/Slaughtered','Production'])
y=df['Yield']

categorical_cols=x.select_dtypes(include=['object','category']).columns.tolist()
num_cols=x.select_dtypes(include=['number']).columns.tolist()

preprocessor=ColumnTransformer(transformers=[
    ("num",Pipeline([
        ("imputer",SimpleImputer(strategy="mean")),
        ("scaler",StandardScaler())
    ]),num_cols),

    ("cat",Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown='ignore'))
    ]),categorical_cols)
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

models={
    "Linear": LinearRegression(),
    "Elastic": ElasticNet(max_iter=5000),
   
}
for name, model in models.items():
    pipeline=Pipeline(steps=[
        ("prepr",preprocessor),
        ("model",model)
    ]) 
    pipeline.fit(x_train,y_train)
    prediction=pipeline.predict(x_test)
    score=r2_score(y_test,prediction)
    mae=mean_absolute_error(y_test,prediction)
    mse=mean_squared_error(y_test,prediction)
    rmse=root_mean_squared_error(y_test,prediction)
    print(
        f"{name}: R2={score:.2f}\n"
        f"MAE={mae:.2f}\n"
        f"MSE={mse:.2f}\n"
        f"RMSE={rmse:.2f}"
          )
    
    
fmodel=Pipeline(steps=[
        ("prepr",preprocessor),
        ("model",LinearRegression())
    ]) 

fmodel.fit(x_train,y_train)
import joblib

joblib.dump(fmodel,"agri_model.pkl")
prediction=fmodel.predict(x_test)

plt.figure(figsize=(10,5))
sns.scatterplot(x=y_test,y=prediction)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"r--")
plt.xlabel('Actual')
plt.ylabel('predicted values')
plt.show()
score=r2_score(y_test,prediction)
mae=mean_absolute_error(y_test,prediction)
mse=mean_squared_error(y_test,prediction)
rmse=root_mean_squared_error(y_test,prediction)
print(
        f"{name}: R2={score:.2f}\n"
        f"MAE={mae:.2f}\n"
        f"MSE={mse:.2f}\n"
        f"RMSE={rmse:.2f}"
          )

print(df['Year'].value_counts())    



