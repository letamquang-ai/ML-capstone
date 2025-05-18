import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pandas_datareader.data as web
import yfinance as yf
from fredapi import Fred

class model():
    def __init__(self):
        self.now=datetime.today().date()
        self.api='eb4ae97ea9d3f7c1a40b6f198b459340'
        self.data=self.take_data_api_key(self.api)
        self.select_now=''
        self.assess={}
        self.models = {
            'Linear Regression': LinearRegression(),
            'Support Vector Machine Linear': SVR(kernel='linear'),
            'Support Vector Machine Quadratic': SVR(kernel='poly', degree=2),
            'Support Vector Machine Cubic': SVR(kernel='poly', degree=3),
            'Support Vector Machine Gaussian': SVR(kernel='rbf', gamma='auto'),
            'Gaussian Process Regression Rational Quadratic': GaussianProcessRegressor(kernel=RationalQuadratic(), alpha=1e-5),
            'Gaussian Process Regression Squared Exponential': GaussianProcessRegressor(kernel=RBF(), alpha=1e-5),
            'Ensemble Learning with Decision Trees Bootstrap Aggregation': BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=10),n_estimators=100,oob_score=True),
            'Ensemble Learning with Decision Trees Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        }
        self.month_dict = {
            "None": 0,
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12
        }
    def take_model(self,name='Linear Regression'):
        return self.models[name]
    def take_price(self,selected_date): 
        if selected_date in self.date.index:
            return self.date[self.date.index==selected_date]['cushing_crude_oil_price'].squeeze()
        return 5.6
    def only_date_price(self):
        return self.date
    def take_future_data(self,days):
        # Copy dữ liệu
        assess_data = self.data.copy()

        # Giới hạn ngày trong assess_data
        assess_data = assess_data[assess_data.index <= self.date.index[-1] - timedelta(days=days)].copy()

        # Dịch index của self.date lùi lại 'days' ngày để khớp với assess_data
        date_shifted = self.date.copy()
        date_shifted.index = date_shifted.index - timedelta(days=days)

        # Join theo index
        assess_data = assess_data.join(date_shifted, how='left', rsuffix='_from_date')
        assess_data.drop(columns=['date','cushing_crude_oil_price_from_date'],inplace=True)
        assess_data.dropna(inplace=True)
        return assess_data

    def predict_for_future_days(self,days:int,model): #need fix
        delta=self.now-datetime.date(self.data.index[-1])
        day=delta.days
        name=self.model_assessment(days,model)
        self.select_now=name
        train=self.take_future_data(day+days)
        test=self.data.iloc[-days:]
        self.models[name].fit(train.drop(labels='cushing_crude_oil_price',inplace=False,axis=1),train['cushing_crude_oil_price'])
        predict=self.models[name].predict(test.drop(labels='cushing_crude_oil_price',inplace=False,axis=1))
        return round((1-self.assess[name])*100,3),pd.DataFrame({'date':pd.date_range(start=self.now+timedelta(days=1),periods=days),'predict':predict})
    def model_assessment(self,days,model):

        assess_data=self.take_future_data(days)

        size=0.7

        train=assess_data.iloc[:int(len(assess_data)*size)]
        test=assess_data.iloc[int(len(assess_data)*size):]

        X_train = train.drop(columns=['cushing_crude_oil_price'])
        y_train = train['cushing_crude_oil_price']
        X_test = test.drop(columns=['cushing_crude_oil_price'])
        y_test = test['cushing_crude_oil_price']

        select=''
        min_score=1
        for nam, mod in self.models.items():
            if model in nam:
                mod.fit(X_train,y_train)
                y_pred=mod.predict(X_test)
                score= np.average(list(map(lambda x,y : np.abs(x-y)/y,y_pred,y_test)))
                self.assess[nam]=score
                if score<min_score:
                    min_score=score
                    select=nam
        return select

    def plot_for_price(self,year, month):
        if year=="None" and self.month_dict[month]==0:
            return self.date
        elif self.month_dict[month]==0:
            return self.date[self.date.index.year == int(year)]
        elif year !="None":
            return self.date[(self.date.index.year == int(year)) & (self.date.index.month == self.month_dict[month])]
        else:
            year=self.now.year
            return self.date[(self.date.index.year == int(year)) & (self.date.index.month == self.month_dict[month])]
    def plot_model(self,name:str,mode:str):
        model_name=name+" "+mode
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(labels='cushing_crude_oil_price',inplace=False,axis=1),self.data['cushing_crude_oil_price'], test_size=0.4, random_state=42)
        self.models[model_name.strip()].fit(X_train, y_train)
        y_pred = self.models[model_name.strip()].predict(X_test)
        plot_data=pd.DataFrame({"Predict":y_pred,"Truth":y_test}).sort_index().reset_index()

        return pd.melt(plot_data, id_vars=['Date'], value_vars=['Predict', 'Truth'], var_name='variable', value_name='value')
    def take_data_api_key(self,api):
        fred = Fred(api_key=api)
        start_date="2003-01-01"
        df=pd.DataFrame()
        df['cushing_crude_oil_price'] = yf.download('CL=F', start_date)['Close'] 
        df['Momentum_5'] = df['cushing_crude_oil_price'].rolling(window=5).apply(lambda x: (np.diff(x) > 0).sum(), raw=True)
        df['Momentum_10'] = df['cushing_crude_oil_price'].rolling(window=10).apply(lambda x: (np.diff(x) > 0).sum(), raw=True)
        df['MA_5'] = df['cushing_crude_oil_price'].rolling(window=5).mean()
        df['MA_10'] = df['cushing_crude_oil_price'].rolling(window=10).mean()
                
        df['dow_jones_adj_close_price'] = yf.download('^DJI', start=start_date)['Close']
        df['nasdaq_adj_close_price'] = fred.get_series('NASDAQCOM', observation_start=start_date)
        df['sp_adj_close_price'] = yf.download('^GSPC', start=start_date)['Close']

        df['usd_to_uer_exchange_rate'] = fred.get_series('DEXUSEU',observation_start=start_date)
        df['usd_to_uk_exchange_rate'] = fred.get_series('DEXUSUK', observation_start=start_date)
        df['jpy_to_usd_exchange_rate'] = fred.get_series('DEXJPUS', observation_start=start_date)

        df['federal_funds_rate'] = fred.get_series('RIFSPFFNB', observation_start=start_date)
        df['bank_prime_loan_rate'] = fred.get_series('DPRIME', observation_start=start_date)
        df['treasury_1_year_rate'] = fred.get_series('DGS1', observation_start=start_date)
        df['treasury_10_year_rate'] = fred.get_series('DGS10', observation_start=start_date)

        df['breakeven_inflation_5_year_rate'] = fred.get_series('T5YIE', observation_start=start_date)
        df['breakeven_inflation_10_year_rate'] = fred.get_series('T10YIE', observation_start=start_date)

        self.date=pd.DataFrame(df['cushing_crude_oil_price'])
        self.date.insert(column='date',value=df.index,loc=0)
        df.dropna(inplace=True)
        for column in df.columns:
            if column in ['dow_jones_adj_close_price','nasdaq_adj_close_price','sp_adj_close_price']:
                df[column] = np.log(df[column])
        return df
# test=model()
# print(test.take_price('2005-04-25'))
    
