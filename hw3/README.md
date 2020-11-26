# Reinforcement learning

## Description:
Simulation of sales, building of a forecasting model, training an agent, comparison of greedy strategy and agent prediction.

## Dataset:
Model sales are going to be a sum of a linear demand function dependent on price, a highly
seasonal component dependent on time with a one-year period; a noise term. We’ll
use google trends data for a seasonal component. csv files with searches in Google.\
You can find detalized description in `homework_3_part_2_description.pdf`. 

## Forecasting
Used `Statsmodel` to get the residuals, trend and seasonality of the data. Predict sales having 4 years in weeks 
in a traning dataframe.

## RL
Created simple agent, who's states are weeks, actions are prices that we have to set on the market. Calculate the reward
 -profit, based on sales, prices and prime costs. Sales depend on price and time period.  

## Author:
[Sofiia Petryshyn](https://github.com/SOFIAshyn/)