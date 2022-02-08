import  pandas as pd
from io import StringIO
import pandas as pd # add to requirments
import numpy as np
from catboost import CatBoostClassifier # add to requirments
from sklearn.preprocessing import PolynomialFeatures # add to requirments
from sklearn.preprocessing import StandardScaler
from io import StringIO
import pickle

data = '28.7967,16.0021,2.6449,0.3918,0.1982,27.7004,22.011,-8.2027,40.092,81.8828'
data = '121.5675,29.0624,3.3412,0.3196,0.1543,-139.7976,-69.814,-26.2826,21.5078,331.8356'
data1 = data.encode('utf-8')
data1 = data1.decode('utf-8')
df = pd.read_csv(StringIO(data1), header=None)

# Загружаем модель и скалер
pt = pickle.load(open('pt_scaler.pkl','rb'))
cat_model = CatBoostClassifier(verbose=False)
cat_model.load_model("cat_model")

df.columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
# EDA
df['fWidth_null'] = 0;
df.loc[df['fWidth'] == 0, 'fWidth_null'] = 1
df['fM3Long_abs'] = np.abs(df['fM3Long'])
df['fM3Trans_abs'] = np.abs(df['fM3Trans'])
# Генерируем фичи
poly = PolynomialFeatures(degree=3)
real_columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist',
                'fM3Long_abs', 'fM3Trans_abs']
df_P_Features = pd.DataFrame(poly.fit_transform(df[real_columns]))
df_P_Features.columns = poly.get_feature_names(real_columns)
df_P_Features.drop(['1'] + real_columns, axis='columns', inplace=True)
df = pd.concat([df, df_P_Features], sort=False, axis=1)
# Скалируем
df_scaled = pd.DataFrame(pt.transform(df))
df_scaled.columns = df.columns
# Предсказываем
preds_class = cat_model.predict(df_scaled)


print(preds_class[0])