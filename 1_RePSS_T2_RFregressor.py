import math
import pickle
import pandas as pd
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def headList(Period,is_gender,is_age,is_bmi,is_hr,is_hrv,is_bp_dia,is_bp_sys):
    head = []
    t = 0
    for i in range(0, Period):
        t = t + 1
        head.append("frame_" + str(t))
    if is_gender:
        head.append("gender")
    if is_age:
        head.append("age")
    if is_bmi:
        head.append("bmi")
    if is_hr:
        head.append("hr")
    if is_hrv:
        head.append("rmssd")
        head.append("sdnn")
        head.append("sdsd")
        head.append("mrri")
        head.append("mhr")
        head.append("sd1")
        head.append("sd2")
    if is_bp_dia:
        head.append('bp_dia')
    if is_bp_sys:
        head.append('bp_sys')
    return head
def MAE(y_true,y_pred):
    length = len(y_true)
    num=0
    print("real  -> predict")
    for i in range(0,length):
        print("%.2f ->% .2f"%(y_true[i],y_pred[i]))
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse,mae
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

txt_path = "data/ascendData_init25.txt"
xlsx_path = "./data/VV-small.xlsx"

names = headList(256,is_gender=False,is_age=True,is_bmi=True,is_hr=True,is_hrv=False,is_bp_dia=True,is_bp_sys=True)
data = pd.read_excel(xlsx_path,sheet_name="support-age15to55-ResAttention")

X = data[names[0:-1]]
Y = data[names[-1:]]

print("feature head:")
print(X.head(1))
print("label head:")
print(Y.head(1))

X = X.replace("?", 0)

x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print("training data size:")
print(len(x_train))
print("testing data size")
print(len(x_test))

#标准化
ss = StandardScaler()
X = ss.fit_transform(X, Y)
x_test = ss.transform(x_test)

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='absolute_error',
                               max_depth=6)
forest.fit(X, Y)

score = forest.score(x_test, y_test)
forest_y_score = forest.predict(x_test)
print ("Accuracy:%.2f%%" % (score * 100))
print(forest_y_score)
print(MAE(y_test.values,forest_y_score))
print(forest.feature_importances_)

with open('VV_bpsys_mae_ResAttention_age_bmi_hr_dia_estimators1000_maxdepth6.pkl','wb') as f:
    pickle.dump(forest,f)

