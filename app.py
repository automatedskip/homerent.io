from flask import Flask, jsonify, render_template, request, make_response
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy
from datetime import datetime
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib

matplotlib.use("Agg")


rent = pd.read_csv("data/city_rent.csv")
mort = pd.read_csv("data/city_home_val.csv", encoding="cp1252")
ts = pd.read_csv("data/Sale_Counts_City.csv")

app = Flask(__name__)


@app.route("/_add_plots")
def add_plots():
    # a = request.args.get("a", 0, type=int)
    # b = request.args.get("b", 0, type=int)
    city = request.args.get("city")
    state = request.args.get("state")
    full_state = request.args.get("full_state")
    # return jsonify(city, state, full_state)

    sep = rent["Location"].str.split(", ", n=1, expand=True)
    rent["City_"] = sep[0]
    rent["State_"] = sep[1]

    rent_c = rent[
        (rent["City_"] == city)
        & (rent["Bedroom_Size"] == "3br")
        & (rent["State"] == state)
    ]

    rent_cm = pd.melt(rent_c)
    rent_cm.drop([0, 1, 2, 3, 78, 79], inplace=True)

    rent_cm["var2"] = rent_cm["variable"].str.replace("_", "-")
    for c in rent_cm["var2"]:
        rent_cm["var4"] = rent_cm["var2"].apply(
            lambda x: x[6:] if x.startswith("P") else x
        )

    rent_cm.drop(["variable", "var2"], axis=1, inplace=True)

    rent_cm["var4"] = pd.to_datetime(rent_cm["var4"])
    rent_cm["value"] = rent_cm["value"].astype(float)

    rent_cm.columns = ["City_Rent_Price", "Date_"]

    mort_c = mort[(mort["RegionName"] == city) & (mort["State"] == state)]

    # Convert the dates that were columns to rows
    # Drop unnecessary rows, Region ID, Region Name,Metro, State, CountyName, SizeRank
    city_df = pd.melt(mort_c)
    city_df = city_df.drop([0, 1, 2, 3, 4, 5])

    # Identify the columns and Update data types
    city_df["variable"] = pd.to_datetime(city_df["variable"])
    city_df["value"] = city_df["value"].astype(float)
    city_df.columns = ["Date_", "City_Home_Price"]

    # Merge the rent dataframe with the home value dataframe
    # Calculate manufactured colulmn Rent_Ratio.
    city_comb = pd.merge(rent_cm, city_df, how="inner", on="Date_")
    city_comb["Rent_Ratio"] = city_comb.City_Home_Price / (
        (city_comb.City_Rent_Price) * 12
    )

    # Work with home sales data set- Need to identify city and state to get unique cities
    cs = ts[(ts["RegionName"] == city) & (ts["StateName"] == full_state)]
    cs1 = pd.melt(cs)
    cs1 = cs1.drop([0, 1, 2, 3])
    cs1["variable"] = pd.to_datetime(cs1["variable"])
    cs1["value"] = cs1["value"].astype(float)
    cs1.columns = ["Date_", "City_Sales"]

    # Merge home sales data with the combined dataframe with historical rent prices,
    # historical home values and calculated historical rent ratio.
    city_ = pd.merge(city_comb, cs1, how="inner", on="Date_")

    ######## PLOT Try
    date1 = city_.Date_.astype("O")
    plt.figure()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax = fig.add_subplot(111)

    ax.plot(date1, city_.Rent_Ratio, color="purple", label="Rent Ratio")
    ax.set(
        ylabel="Rent Ratio",
        title=str(city) + "," + str(state) + " City home sales and Rent ratio",
    )
    ax.legend(loc=4)
    ax2 = ax.twinx()
    ax2.plot(date1, city_.City_Sales, color="gold", label="Home Sales")
    ax2.set(ylabel="Home Sales")
    ax2.legend(loc=2)

    figfile = io.BytesIO()
    plt.savefig(figfile, format="png")
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = figfile.getvalue()  # extract string (stream of bytes)
    import base64

    figdata_png = base64.b64encode(figdata_png)
    figdata_str = figdata_png.decode("utf-8")
    # print(figdata_png.decode("utf-8"))
    # return jsonify(result=figdata_str)

    ############################################################################

    # Make 2 copies of city_ for forecasting of Rent Ratio and Home sales
    df_sales = city_.copy()
    df_ratio = city_.copy()

    # add previous sales to the next row
    df_sales["prev_sales"] = df_sales["City_Sales"].shift(1)
    df_ratio["prev_ratio"] = df_ratio["Rent_Ratio"].shift(1)

    # Since both Home Sales and rent ratio were not stationary- take the difference between current month sales and
    # Prior month for "diff" column.   Also done for rent ratio, to get the column "diff_r"
    df_sales["diff"] = df_sales["City_Sales"] - df_sales["prev_sales"]
    df_ratio["diff_r"] = df_ratio["Rent_Ratio"] - df_ratio["prev_ratio"]
    #  ELIMINIATED CODE FOR STATIONARY PLOTS OF DIFF and DIFF_R

    ## Build Supervised value with 48 lags - (2 yrs forecasts)-
    # data frames for model based from our sales dataframe and rent ratio data frames

    df_s = df_sales.drop(["prev_sales"], axis=1)
    df_r = df_ratio.drop(["prev_ratio"], axis=1)

    # Use the previous monthly sales data to forecast the next ones.  Using a lookback period of
    # 2 yrs (48 months)

    for d in range(1, 49):
        field_name = "lag_" + str(d)
        df_r[field_name] = df_r["diff_r"].shift(d)
    # drop null values
    df_r = df_r.dropna().reset_index(drop=True)

    for v in range(1, 49):
        field_name = "lag_" + str(v)
        df_s[field_name] = df_s["diff"].shift(v)
    # drop null values
    df_s = df_s.dropna().reset_index(drop=True)

    import statsmodels.formula.api as smf

    # Define the regression formula
    model_s = smf.ols(
        formula="diff ~ lag_1+lag_2+lag_3+lag_4+lag_5+lag_6+lag_7\
    +lag_8+lag_9+lag_10+lag_11+lag_12+lag_13+lag_14+lag_15+lag_16\
    +lag_17+lag_18+lag_19+lag_20+lag_21",
        data=df_s,
    )
    # Fit the regression
    model_fit_s = model_s.fit()
    # Extract the adjusted r-squared
    # regression_adj_rsq_s = model_fit_s.rsquared_adj
    # print(regression_adj_rsq_s)

    # From the statsmodels.formula.api as smf
    # Define the regression formula
    model_r = smf.ols(formula="diff_r ~ lag_1+lag_2+lag_3+lag_4+lag_5+lag_6", data=df_r)
    # Fit the regression
    model_fit_r = model_r.fit()
    # Extract the adjusted r-squared
    regression_adj_rsq_r = model_fit_r.rsquared_adj
    # print(regression_adj_rsq_r)

    from sklearn.preprocessing import MinMaxScaler

    df_s_model = df_s.drop(
        ["City_Sales", "Date_", "City_Rent_Price", "City_Home_Price", "Rent_Ratio"],
        axis=1,
    )
    # split train and test set
    train_dfs, test_dfs = df_s_model[0:-12].values, df_s_model[-12:].values

    df_r_model = df_r.drop(
        ["City_Sales", "Date_", "City_Rent_Price", "City_Home_Price", "diff_r"], axis=1
    )
    # split train and test set
    train_set2, test_set2 = df_r_model[0:-12].values, df_r_model[-12:].values

    ## SCALE DATA
    from sklearn.preprocessing import MinMaxScaler

    scaler_dfs = MinMaxScaler(feature_range=(-1, 1))
    scaler_dfs = scaler_dfs.fit(train_dfs)

    # reshape training set
    train_dfs = train_dfs.reshape(train_dfs.shape[0], train_dfs.shape[1])
    train_dfs_scaled = scaler_dfs.transform(train_dfs)
    # # reshape test set
    test_dfs = test_dfs.reshape(test_dfs.shape[0], test_dfs.shape[1])
    test_dfs_scaled = scaler_dfs.transform(test_dfs)

    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    scaler2 = scaler2.fit(train_set2)
    # reshape training set
    train_set2 = train_set2.reshape(train_set2.shape[0], train_set2.shape[1])
    train_set_scaled2 = scaler2.transform(train_set2)

    # reshape test set
    test_set2 = test_set2.reshape(test_set2.shape[0], test_set2.shape[1])
    test_set_scaled2 = scaler2.transform(test_set2)

    import sklearn
    import keras
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    from keras.utils import np_utils
    from keras.layers import LSTM
    from keras.layers import Dropout
    from sklearn.model_selection import KFold, cross_val_score, train_test_split

    epochs1 = 20

    # create feature and label sets from scaled datasets:
    X_train3, y_train3 = train_dfs_scaled[:, 1:], train_dfs_scaled[:, 0:1]
    X_train3 = X_train3.reshape(X_train3.shape[0], 1, X_train3.shape[1])
    X_test3, y_test3 = test_dfs_scaled[:, 1:], test_dfs_scaled[:, 0:1]
    X_test3 = X_test3.reshape(X_test3.shape[0], 1, X_test3.shape[1])

    X_train2, y_train2 = train_set_scaled2[:, 1:], train_set_scaled2[:, 0:1]
    X_train2 = X_train2.reshape(X_train2.shape[0], 1, X_train2.shape[1])
    X_test2, y_test2 = test_set_scaled2[:, 1:], test_set_scaled2[:, 0:1]
    X_test2 = X_test2.reshape(X_test2.shape[0], 1, X_test2.shape[1])

    ## Build LSTM Network

    # State sales forecast Model
    model_3 = Sequential()
    model_3.add(
        LSTM(
            4,
            batch_input_shape=(1, X_train3.shape[1], X_train3.shape[2]),
            stateful=True,
        )
    )
    model_3.add(Dense(1))
    model_3.add(Dropout(0.2))
    model_3.compile(loss="mean_squared_error", optimizer="adam")
    model_3.fit(
        X_train3, y_train3, epochs=epochs1, batch_size=1, verbose=1, shuffle=False
    )

    ## Setup for history reference for loss plot
    his = model_3.fit(
        X_train3,
        y_train3,
        epochs=epochs1,
        batch_size=1,
        verbose=1,
        shuffle=False,
        validation_data=(X_test3, y_test3),
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
    )

    model_3.summary()

    # Rent ration forecast Model
    model2 = Sequential()
    model2.add(
        LSTM(
            4,
            batch_input_shape=(1, X_train2.shape[1], X_train2.shape[2]),
            stateful=True,
        )
    )
    model2.add(Dense(1))
    model2.add(Dropout(0.2))
    model2.compile(loss="mean_squared_error", optimizer="adam")
    model2.fit(
        X_train2, y_train2, epochs=epochs1, batch_size=1, verbose=1, shuffle=False
    )
    ## Setup for history reference for loss plot
    hist = model2.fit(
        X_train2,
        y_train2,
        epochs=epochs1,
        batch_size=1,
        verbose=1,
        shuffle=False,
        validation_data=(X_test2, y_test2),
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
    )
    model2.summary()

    ## Plot loss for each model - ELIMINIATED PLOTS

    # Make predictions

    y_pred3 = model_3.predict(X_test3, batch_size=1)
    y_pred2 = model2.predict(X_test2, batch_size=1)

    # Perform Inverse transform for scaled data
    y_pred2 = y_pred2.reshape(y_pred2.shape[0], 1, y_pred2.shape[1])

    # rebuild test set for inverse transform
    pred_test_set2 = []
    for index in range(0, len(y_pred2)):
        # print (numpy.concatenate([y_pred2[index],X_test2[index]],axis=1))
        pred_test_set2.append(
            numpy.concatenate([y_pred2[index], X_test2[index]], axis=1)
        )

    # reshape pred_test_set
    pred_test_set2 = numpy.array(pred_test_set2)
    pred_test_set2 = pred_test_set2.reshape(
        pred_test_set2.shape[0], pred_test_set2.shape[2]
    )

    # inverse transform
    pred_test_set_inverted2 = scaler2.inverse_transform(pred_test_set2)

    # ---------------------------------------------------------------------------------------------
    y_pred3 = y_pred3.reshape(y_pred3.shape[0], 1, y_pred3.shape[1])
    # rebuild test set for inverse transform
    pred_test_dfs = []
    for x in range(0, len(y_pred3)):
        # print (numpy.concatenate([y_pred3[x],X_test3[x]],axis=1))
        pred_test_dfs.append(numpy.concatenate([y_pred3[x], X_test3[x]], axis=1))

    # reshape pred_test_dfs
    pred_test_dfs = numpy.array(pred_test_dfs)
    pred_test_dfs = pred_test_dfs.reshape(
        pred_test_dfs.shape[0], pred_test_dfs.shape[2]
    )

    # inverse transform
    pred_test_dfs_inverted = scaler_dfs.inverse_transform(pred_test_dfs)
    # print(pred_test_dfs_inverted.shape)

    # Build the dataframe that has the dates and the predictions.
    # Transformed predictions are showing the difference.
    # Calculate the predicted sales numbers along with rent ratio for 12 months.

    result_list_3 = []
    sales_dates_3 = list(df_sales[-13:].Date_)
    act_sales_3 = list(df_sales[-13:].City_Sales)
    for index in range(0, len(pred_test_dfs_inverted)):
        result_dict_3 = {}
        result_dict_3["pred_value"] = int(
            pred_test_dfs_inverted[index][0] + act_sales_3[index]
        )
        result_dict_3["Date_"] = sales_dates_3[index + 1]
        result_list_3.append(result_dict_3)
    df_result_3 = pd.DataFrame(result_list_3)
    df_result_3

    # ---------------------------------------------------------------------------------------------

    result_list2 = []
    ratio_dates = list(df_ratio[-13:].Date_)
    act_ratio = list(df_ratio[-13:].Rent_Ratio)
    for index2 in range(0, len(pred_test_set_inverted2)):
        result_dict2 = {}
        result_dict2["pred_ratio"] = (
            pred_test_set_inverted2[index2][0] + act_ratio[index2]
        )
        result_dict2["Date_"] = ratio_dates[index2 + 1]
        result_list2.append(result_dict2)
    df_result2 = pd.DataFrame(result_list2)

    new_df2 = pd.merge(df_ratio, df_result2, how="left", on="Date_")
    new_df3 = pd.merge(df_sales, df_result_3, how="left", on="Date_")

    ## Plot how model works with test data for the past 12 months

    date3 = new_df3.Date_.astype("O")
    date2 = new_df2.Date_.astype("O")

    #### ELIMINATED PLOTS  #######################################

    # Forcasting for next 12 months for city sales and rent ratio
    dfs_1 = df_result_3.copy()
    dfs_1 = dfs_1.set_index("Date_")

    dfs_2 = df_result2.copy()
    dfs_2 = dfs_2.set_index("Date_")

    from pandas.tseries.offsets import DateOffset

    add_dates_3 = [dfs_1.index[-1] + DateOffset(month=x) for x in range(0, 13)]
    future_dates_3 = pd.DataFrame(index=add_dates_3[1:], columns=dfs_1.columns)

    add_dates_4 = [dfs_2.index[-1] + DateOffset(month=x) for x in range(0, 13)]
    future_dates_4 = pd.DataFrame(index=add_dates_4[1:], columns=dfs_2.columns)

    # Reshape the predictions and move to list for inverse transform purposes
    yp = y_pred3.reshape(-1, 1)
    ypp = yp.tolist()

    yp2 = y_pred2.reshape(-1, 1)
    yp2_l = yp2.tolist()

    train_set2_ = train_set2.copy()
    ytrain_ratio = train_set2_[:, 0:1]

    train_dfs_ = train_dfs.copy()
    ytrain_city = train_dfs_[:, 0:1]

    # Scale the ytrain data set and retrieve the inverse transforms of predicted values
    # For Rent Ratio and Home Sales

    scaler_t = MinMaxScaler()
    scaler_t.fit(ytrain_city)

    scaler_tt = MinMaxScaler()
    scaler_tt.fit(ytrain_ratio)

    df_predict3 = pd.DataFrame(
        scaler_t.inverse_transform(ypp),
        index=future_dates_3[-13:].index,
        columns=["Prediction"],
    )

    df_predict4 = pd.DataFrame(
        scaler_tt.inverse_transform(yp2_l),
        index=future_dates_4[-13:].index,
        columns=["Ratio_Prediction"],
    )

    df_predict3.index.names = ["Date_"]
    df_predict4.index.names = ["Date_"]

    df_22 = df_sales[["Date_", "City_Sales"]]
    df_44 = df_ratio[["Date_", "Rent_Ratio"]]

    df_cs = pd.merge(df_22, df_predict3, how="outer", on="Date_")
    df_rr = pd.merge(df_44, df_predict4, how="outer", on="Date_")

    ## Plot figures for prediction for rent ratio and home sales

    # plt.figure()
    plt.figure(figsize=(10, 4))
    plt.plot(df_cs.Date_, df_cs.City_Sales, color="b", label="Historical Home Sales")
    plt.plot(
        df_cs.Date_,
        (df_cs.Prediction) * -2.78,
        color="b",
        label="Forecasted Home Sales",
        ls="--",
    )
    plt.legend()
    plt.title(str(city) + "," + str(state) + " Home sales historical and forecasted")

    figfile2 = io.BytesIO()
    plt.savefig(figfile2, format="png")
    figfile2.seek(0)  # rewind to beginning of file
    figdata_png2 = figfile2.getvalue()  # extract string (stream of bytes)
    import base64

    figdata_png2 = base64.b64encode(figdata_png2)
    figdata_str2 = figdata_png2.decode("utf-8")
    # print(figdata_png2.decode("utf-8"))

    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        df_cs.Date_,
        (df_cs.Prediction) * -2.78,
        color="blue",
        ls="--",
        label="Homes Sales Forecast",
    )
    ax2 = ax.twinx()
    ax2.plot(
        df_cs.Date_,
        (df_rr.Ratio_Prediction) * 1.075,
        color="red",
        ls="--",
        label="Rent Ratio Forecast",
    )
    # ax.set_ylabel('Rent Ratio')
    ax.legend(loc=4)
    ax.set(ylabel="Home sales forecast")
    ax.set(
        title=str(city)
        + ","
        + str(state)
        + " Forecasted city home sales and rent ratio"
    )
    ax2.set(ylabel="Forecasted Rent Ratio")
    ax2.legend(loc="best")

    figfile3 = io.BytesIO()
    plt.savefig(figfile3, format="png")
    figfile3.seek(0)
    figdata_png3 = figfile3.getvalue()

    figdata_png3 = base64.b64encode(figdata_png3)
    figdata_str3 = figdata_png3.decode("utf-8")
    # print(figdata_png2.decode("utf-8"))

    # Box plots of home sales and rent ratio per Quarter
    dfa = city_.copy()
    dfa["quarter"] = dfa["Date_"].apply(lambda x: x.quarter)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.subplots_adjust(wspace=0.2)
    sns.boxplot(x="quarter", y="City_Sales", data=dfa)
    plt.xlabel("Quarter")
    plt.title("Box plot of Quarterly City Sales: " + str(city) + "," + str(state))
    sns.despine(left=True)
    plt.tight_layout()

    figfile4 = io.BytesIO()
    plt.savefig(figfile4, format="png")
    figfile4.seek(0)
    figdata_png4 = figfile4.getvalue()

    figdata_png4 = base64.b64encode(figdata_png4)
    figdata_str4 = figdata_png4.decode("utf-8")

    return jsonify(
        result=figdata_str,
        result2=figdata_str2,
        result3=figdata_str3,
        result4=figdata_str4,
    )


"""


    fig, ax = plt.subplots(figsize=(10, 5))

    ax = fig.add_subplot(111)

    ax.plot(date1, city_.Rent_Ratio, color="purple", label="Rent Ratio")
    ax.set(
        ylabel="Rent Ratio",
        title=str(city) + "," + str(state) + " City home sales and Rent ratio",
    )
    ax.legend(loc=4)
    ax2 = ax.twinx()
    ax2.plot(date1, city_.City_Sales, color="gold", label="Home Sales")
    ax2.set(ylabel="Home Sales")
    ax2.legend(loc=2)

    figfile = io.BytesIO()
    plt.savefig(figfile, format="png")
    figfile.seek(0)
    figdata_png = figfile.getvalue()
    import base64

    figdata_png = base64.b64encode(figdata_png)

    import simplejson as json

    figdata = json.dumps(list(figdata_png))

    canvas = FigureCanvas(fig)
    png_output = io.BytesIO()
    canvas.print_png(png_output)
    png_string = base64.b64encode(png_output.read()).decode("utf-8")
    # print(figdata_png)
    return jsonify(result=figdata)
    # result = figdata
    # return result 

"""


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":

    app.run(host="127.0.0.1", debug=False, use_reloader=True, threaded=False)
