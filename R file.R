#Time Series analysis and forecasting for Frontier airline flight delays :

# Libraries used in this time series forecasting
# install.packages("forecast")
# install.packages("zoo")
library(forecast)
library(zoo)

# Creating a dataframe for the flight delays data 

flights.data <- read.csv('monthly-flights.csv',header= TRUE)

# To see the first 6 and last 6 records of the data set

head(flights.data)
tail(flights.data)

# Creating Time Series dataset for flight delays data
# The timeseries generated for the Departure_delay column only
# The timeseries data generated starting from Jan/2008 
# The timeseries data generated untill Dec/2019
# The parameter frequency will be 12 as it is a monthly data

delays.ts <- ts(flights.data$Departure_delay, start = c(2008, 1), end = c(2019, 12), freq = 12)


# Using the plot() to plot flight delays timeseries data 

plot(delays.ts, 
     xlab = "Time", ylab = "Monthly Airline Delays (in hours.minutes)", ylim = c(100, 300), bty = "l",
     xaxt = "n", xlim = c(2008, 2020.25), main = "Frontier Airlines - Flight Delays", lwd = 3, col="blue") 
axis(1, at = seq(2008, 2020.25, 1), labels = format(seq(2008, 2020.25, 1)))

# stl() function to plot times series components of the original data. 

flightdelays.stl <- stl(delays.ts, s.window = "periodic")
autoplot(flightdelays.stl, main = "Flight delays Time Series Components")

# Acf() function to identify autocorrealtion and to plot autocorrrelation for different lags

autocor <- Acf(delays.ts, lag.max = 12, main = "Autocorrelation for Frontier Airlines Delays")


# Dataframe for autocorrelation coefficients for various lags in the flight delays timeseries data

Lag <- round(autocor$lag, 0)
ACF <- round(autocor$acf, 3)
data.frame(Lag, ACF)

# Partitioning the entire flight delays dataset into 
# training (96 data periods) and validation(48 data periods )

nValid <- 48 
nTrain <- length(delays.ts) - nValid
#nTrain
train.ts <- window(delays.ts, start = c(2008, 1), end = c(2008, nTrain)) 
#train.ts
valid.ts <- window(delays.ts, start = c(2008, nTrain + 1), 
                   end = c(2008, nTrain + nValid))
#valid.ts

# Using the plot () to visualize data partitions. 

plot(train.ts, 
     xlab = "Time", ylab = "Monthly Airline Delays (in hours.minutes)", ylim = c(100,300), bty = "l",
     xaxt = "n", xlim = c(2008, 2022.25), main = "graph", lwd = 2,col = "blue") 
axis(1, at = seq(2008, 2022, 1), labels = format(seq(2008, 2022, 1)))
lines(valid.ts, col = "black", lty = 1, lwd = 2)

# Vertical lines and horizontal arrows describing training, validation, and future prediction intervals.

lines(c(2019 - 3, 2019 - 3), c(0, 2600)) 
lines(c(2019.92, 2019.92), c(0, 2600)) 
text(2012, 300, "Training")
text(2017.45, 300, "Validation")
text(2021, 300, "Future")

arrows(2016, 280, 2007.5, 280, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2016.10, 280, 2019.89, 280, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2020.10, 280, 2022.75, 280, code = 3, length = 0.1,
       lwd = 1, angle = 30)


####----- MODEL GENERATION-----####

# REGRESSION MODELS

# i.	Regression model with linear trend and seasonality:

train.lineartrend.season <- tslm(train.ts ~ trend + season)
summary(train.lineartrend.season)
 
# Forecast for validation period

train.lineartrend.season.pred <- forecast(train.lineartrend.season, h = nValid, level = 0)
train.lineartrend.season.pred

# ii.	Regression model with quadratic trend and seasonality

train.quadtrend.season <- tslm(train.ts ~ trend + I(trend^2) + season)
summary(train.quadtrend.season)

# Forecast for validation period

train.quadtrend.season.pred <- forecast(train.quadtrend.season, h = nValid, level = 0)
train.quadtrend.season.pred

## Using the accuracy() to compare accuracy of the two regressions models developed

round(accuracy(train.lineartrend.season.pred, valid.ts), 3)
round(accuracy(train.quadtrend.season.pred, valid.ts), 3)

# Based on the accuracy() developing 
# REGRESSION MODEL WITH LINEAR TREND AND SEASONALITY USING ENTIRE DATASET

entiredata.lineartrend.season <- tslm(delays.ts ~ trend + season)
summary(entiredata.lineartrend.season)

entiredata.lineartrend.season.pred <- forecast(entiredata.lineartrend.season, h = 12, level = 0)
entiredata.lineartrend.season.pred

# Comparing the accuracy of the above developed regression model along with
# the baseline models like naive model, seasonal naive

round(accuracy(entiredata.lineartrend.season.pred$fitted, delays.ts), 3)
round(accuracy((naive(delays.ts))$fitted, delays.ts), 3)
round(accuracy((snaive(delays.ts))$fitted, delays.ts), 3)


## VIZUALIZATION OF THE MODEL PREDICTIONS vs. ACTUAL DATA

plot(delays.ts, 
     xlab = "Time", ylab = "Ridership (in 000s)", ylim = c(100, 400), bty = "l",
     xlim = c(2008, 2021.25), lwd = 2, col = "black",
     main = "Regression Model with Linear Trend and Seasonality and Forecast for Future Periods", 
     flty = 5) 
axis(1, at = seq(2008, 2021, 1), labels = format(seq(2008, 2021, 1)))
lines(entiredata.lineartrend.season$fitted, col = "green", lwd = 2)
lines(entiredata.lineartrend.season.pred$mean, col = "blue", lty = 5, lwd = 2)
legend(2008,380, legend = c("Flight Delays Time Series", 
                            "Regression Model with Linear Trend and Seasonality for Entire Data",
                            "Regression Model with Linear Trend and Seasonality Forecast for Future 12 Periods"), 
       col = c("black", "green" , "blue"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")

lines(c(2020, 2020), c(0, 400))
text(2014, 400, "Entire Data")
text(2020.5, 400, "Future")
arrows(2008.0 - 0.5, 390, 2020, 390, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2020, 390, 2021, 390, code = 3, length = 0.1,
       lwd = 1, angle = 30)

## RESIDUALS PLOT 

plot(entiredata.lineartrend.season.pred$residuals, 
     xlab = "Time", ylab = "Residuals", ylim = c(-400, 400), bty = "l",
     xlim = c(2008, 2021.25), lwd = 2, col = "brown",
     main = "Residuals for Regression Model with Linear Trend and Seasonality Forecast for entire data", 
     flty = 5)

axis(1, at = seq(2008, 2021, 1), labels = format(seq(2008, 2021, 1)))

lines(c(2020, 2020), c(-400, 400))
text(2014, 350, "Entire Data")
text(2020.5, 350, "Future")
arrows(2008.0 - 0.5, 280, 2020, 280, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2020, 280, 2021, 280, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#---------------------------------------------------------------#

# TWO-LEVEL MODEL GENERATION

# Developing two-level model for entire dataset

# i. Regression model with linear trend and seasonality

entiredata.lineartrend.season <- tslm(delays.ts ~ trend + season)
summary(entiredata.lineartrend.season)


entiredata.lineartrend.season.pred <- forecast(entiredata.lineartrend.season, h = 12, level = 0)
entiredata.lineartrend.season.pred

# To identify and display residulas for time series based on the regression

entiredata.lineartrend.season.res <- entiredata.lineartrend.season$residuals
entiredata.lineartrend.season.res  

# Apply trailing MA with 12 periods in the window to residuals

ma.trailing.res_12 <- rollmean(entiredata.lineartrend.season.res , k = 12, align = "right")
ma.trailing.res_12

# Forecast for residuals 12 periods into the future

ma.trailing.res_12.pred <- forecast(ma.trailing.res_12, h = 12, level = 0)
ma.trailing.res_12.pred 

# Developing real forecast for 12 periods into the future 
# by combining regression forecast and trailing MA forecast for residuals.

ts.forecast.12 <- entiredata.lineartrend.season.pred$mean + ma.trailing.res_12.pred$mean
ts.forecast.12


# Table with regression forecast, trailing MA for residuals and total forecast for 12 months into the future with two level data

total.reg.ma.pred <- data.frame(entiredata.lineartrend.season.pred$mean, ma.trailing.res_12.pred$mean, 
                                ts.forecast.12)
total.reg.ma.pred

# accuracy() function to identify common accuracy measures for the developed models

round(accuracy(entiredata.lineartrend.season.pred$fitted, delays.ts), 3)
round(accuracy(entiredata.lineartrend.season.pred$fitted+ma.trailing.res_12, delays.ts), 3)


# ii. Developing the Holt-Winter's model on the entire flights delay dataset

delays.HW.ZZZ <- ets(delays.ts , model = "ZZZ")
delays.HW.ZZZ 

# Developing real forecast for 12 periods into the future

flightpred.HW.ZZZ <- forecast(delays.HW.ZZZ, h = 12 , level = 0)
flightpred.HW.ZZZ

# accuracy() function to identify common accuracy measures for the developed models

round(accuracy(flightpred.HW.ZZZ$fitted, delays.ts), 3)
round(accuracy(entiredata.lineartrend.season.pred$fitted+ma.trailing.res_12, delays.ts), 3)

## plot()  to generate plot for Original Data , Regression forecast and 
#  Predictions  into 12 periods into the future

# Plot for original Ridership time series data and regression model developed

plot(delays.ts, 
     xlab = "Time", ylab = "Monthly Airline Delays (in hours.minutes)", ylim = c(100,300), bty = "l",
     xaxt = "n", xlim = c(2008, 2020.25), lwd =2,
     main = "Flight Delays and Regression with Trend and Seasonality") 
axis(1, at = seq(2008, 2020.25, 1), labels = format(seq(2008, 2020.25, 1)))
lines(delays_reg_seas$fitted, col = "brown", lwd = 2)
lines(delays_reg_seas_pred$mean, col = "brown", lty =5, lwd = 2)
legend(2009,300, legend = c("Flight-delays", "Regression",
                            "Regression Forecast for 12 Periods into Future"), 
       col = c("black", "brown" , "brown"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")


# Plot to show regression residuals data and trailingMA based on residuals

plot(delays_reg_seas_res, 
     xlab = "Time", ylab = "Monthly Airline Delays (in hours.minutes)", ylim = c(-10, 10), bty = "l",
     xaxt = "n", xlim = c(2008, 2020.25), lwd =2, 
     main = "Regression Residuals and Trailing MA for Residuals, k =12") 
axis(1, at = seq(2008, 2020.25, 1), labels = format(seq(2008, 2020.25, 1)))
lines(ma.trailing.res_12, col = "blue", lwd = 2, lty = 1)
lines(ma.trailing.res_12.pred$mean, col = "red", lwd = 2, lty = 5)
legend(2009, 12, legend = c("Regresssion Residuals", "Trailing MA for Residuals, k=12", 
                            "Trailing MA Forecast for 12 Periods into Future"), 
       col = c("black", "blue", "red"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")

#------------------------------------------------------------------#

# ARIMA MODEL GENERATION

# Uisng AR(1) model to find historical candy data is predictable or not

delays.ts.ar1<- Arima(delays.ts, order = c(1,0,0))
summary(delays.ts.ar1)

# Using differencing and Acf() function to find historical candy data is predictable or not

diff.delays.ts <- diff(delays.ts, lag = 1)
diff.delays.ts

Acf(diff.delays.ts, lag.max = 12, 
    main = "Autocorrelation for differenced Monthly Airline Delays")


# i.	Regression model with linear trend and seasonality:

train.lineartrend.season <- tslm(train.ts ~ trend + season)
summary(train.lineartrend.season)

# Forecast for validation period

train.lineartrend.season.pred <- forecast(train.lineartrend.season, h = nValid, level = 0)
train.lineartrend.season.pred

# Use Acf() function to identify autocorrealtion for the model residuals (training set)
# to alos plot autocorrrelation for different lags

Acf(train.lineartrend.season.pred$residuals, lag.max = 12, 
    main = "Autocorrelation for Monthly Airline Delays's Training Residuals")


# Arima() function to fit AR(1) model for training residulas

res.ar1 <- Arima(train.lineartrend.season$residuals, order = c(1,0,0))
summary(res.ar1)

# forecast() function is used to make prediction of residuals in validation set

res.ar1.pred <- forecast(res.ar1, h = nValid, level = 0)
res.ar1.pred

# Acf() function to identify autocorrealtion for the training residual of residuals 
# and plot autocorrrelation for different lags

Acf(res.ar1$residuals, lag.max = 12, 
    main = "Autocorrelation for Monthly Airline Delays's Training Residuals of Residuals")


# Creating two-level model, regression + AR(1) for validation period
# Data table with historical validation data, regression forecast for validation period
# AR(1) for validation, and and two level model results. 

valid.two.level.pred <- train.lineartrend.season.pred$mean + res.ar1.pred$mean

valid.df <- data.frame(valid.ts, train.lineartrend.season.pred$mean, 
                       res.ar1.pred$mean, valid.two.level.pred)

names(valid.df) <- c("Valid.Production", "Reg.Forecast", 
                     "AR(1)Forecast", "Combined.Forecast")

valid.df

# Entire dataset

entiredata.lineartrend.season <- tslm(delays.ts ~ trend + season)
summary(entiredata.lineartrend.season)

entiredata.lineartrend.season.pred <- forecast(entiredata.lineartrend.season, h = 12, level = 0)
entiredata.lineartrend.season.pred


# Arima() function to fit AR(1) model for regression residulas

residual.ar1 <- Arima(entiredata.lineartrend.season$residuals, order = c(1,0,0))
summary(residual.ar1)

# forecast() function to make prediction of residuals into the future 12 months.

residual.ar1.pred <- forecast(residual.ar1, h = 12, level = 0)

# Acf() function to identify autocorrealtion for the residual of residuals 
# and to plot autocorrrelation for different lags 

Acf(residual.ar1$residuals, lag.max = 12, 
    main = "Autocorrelation for AR(1) Model Residuals for Entire Data Set")

# To Iidentify forecast for the future 12 periods as sum of linear trend and seasonal model
# and AR(1) model for residuals

trend.season.ar1.pred <- entiredata.lineartrend.season.pred$mean + residual.ar1.pred$mean
trend.season.ar1.pred

# Data table with linear trend and seasonal forecast for 12 future periods,
# AR(1) model for residuals for 12 future periods, and combined two-level forecast for
# 12 future periods

table.df <- data.frame(entiredata.lineartrend.season.pred$mean, 
                       residual.ar1.pred$mean, trend.season.ar1.pred)

names(table.df) <- c("Reg.Forecast", "AR(1)Forecast","Combined.Forecast")

table.df

# accuracy() function to identify common accuracy measures for the developed models

round(accuracy(entiredata.lineartrend.season$fitted + residual.ar1$fitted, delays.ts), 3)
round(accuracy(train.auto.arima.pred$fitted, delays.ts), 3)
