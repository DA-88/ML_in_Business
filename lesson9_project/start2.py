import flask # add to requirments
import pandas as pd # add to requirments
import numpy as np
import xgboost as xgb # add to requirments
from sklearn.preprocessing import PolynomialFeatures # add to requirments
from sklearn.preprocessing import StandardScaler
from io import StringIO
import pickle

# Загружаем модель и скалер
pt = pickle.load(open('pt_scaler.pkl','rb'))
bst = xgb.XGBClassifier()
bst.load_model('xgb_model')

# Иницируем flask
print(flask.__version__)
print(__name__)
app = flask.Flask(__name__)

@app.route("/")
def index():
    return 'test success'

@app.route("/predict", methods=["POST"])
def return_values():
    answ = calc_res(flask.request.get_data())
    return answ

def calc_res(data):
    data = data.decode('utf-8')
    df = pd.read_csv(StringIO(data), header=None)
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
    preds_class = bst.predict(df_scaled)
    res = '0'
    if preds_class[0] == 1:
        res = 'gamma (signal)'
    else:
        res = 'hadron (background)'

    return res



app.run(host='0.0.0.0', port=8180, debug=False)