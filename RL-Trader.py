# Reinforcement Learning Equity Trader
# By Matija Krolo || github.com/Kroat
# To use:
# 1.) Ensure all necessary libraries are installed on your machine (Numpy, Scipy, Pandas)
# 2.) Place RL-Trader.py in a project directory
# 3.) cd into the project directory
# 4.) Pass the command:
#       RL-Trader.py [EQUITY] [START DATE - DAY/MONTH/YEAR] [STARTING PORTFOLIO VALUE] [HOW MANY TRADES TO RUN BEFORE REINFORCEMENT LEARNING BEGINS]
#
# ex: RL-Trader.py F 1/1/2000 1000 200
# ^ This runs RL-Trader against Ford's historical data and analyzes 200 trades before the Reinforcement Learning begins with a beginning portfolio of $1,000
# ** Please note that this RL script only trades one equity at a time!

# Edit these values to change how the RL brain learns
EPSILON = .5
EPSILON_DECAY = 0.1
ALPHA = .01
GAMMA = .8
priceAtPurchase = 0

# Create agent class
class Agent:
    def __init__(self, alpha_input, epsilon_input, gamma_input):
        self.alpha = alpha_input
        self.epsilon = 1 - epsilon_input
        self.gamma = gamma_input

# Create class object
agent = Agent(ALPHA,EPSILON, GAMMA)

# Import Libraries
import numpy as np
from scipy import stats
from math import log
import pandas as pd
import sys, time, datetime
from Logic.logic import calculate_BSM, state_logic

import get_data

# Get passed-in arguments
GIVEN_EQUITY, START_DATE, STARTING_PORTFOLIO_VALUE, TRADES_TO_RUN = 'AAPL','2000-01-01', 5000, 4500#sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# Error check arguments
# if len(sys.argv) != 5:
#     #print("To run: RL-Trader.py [EQUITY] [START DATE - DAY/MONTH/YEAR] [STARTING PORTFOLIO VALUE] [HOW MANY TRADES TO RUN BEFORE REINFORCEMENT LEARNING BEGINS]\nEx. RL-Trader.py F 1/1/2000 1000 200")
#     exit()

# Get Equity Data
CURRENT_MONTH = datetime.datetime
# Todo: create datetime function for user inputs on end dates
EQUITY = get_data.dataset_loader(GIVEN_EQUITY, end='2019-01-01', start=START_DATE,interval='1d')
MKT_VOLATIILTY =  get_data.dataset_loader('^VIX', end='2019-01-01', start=START_DATE, interval='1d')
RF_Rate = get_data.dataset_loader('^TNX', end='2019-01-01', start=START_DATE,interval='1d')

# Don't edit these
STATES = 3
# Actions of Q-Table
ACTIONS = ['buy', 'sell']
# Holds total trades that can be made
TOTAL_TRADES = len(EQUITY['Close']) 

# Error Check
if int(TRADES_TO_RUN) > TOTAL_TRADES:
    print("\nThere are only " + str(TOTAL_TRADES) + " trading days available from data, which is greater than the input of " + str(TRADES_TO_RUN) + ". Please try again.")
    exit()

# Q-Table generator function
def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))),
                         columns=actions)
    return table

# Create dictionary
compile_data = {'EQUITY': EQUITY['Close'], 'RF': RF_Rate['Close'
                ], 'SIGMA': MKT_VOLATIILTY['Close']}

# Compile dataframe from dictionary
data = pd.DataFrame(compile_data)

# Agent brain for RL
def choose_trade(pointer, q_table):
    # Logic is only running
    # if pointer < int(TRADES_TO_RUN):
    #     print ("Reinforcement Learning not initiated yet, Q-Table still building.")
    # Find the trade decision from our trade logic
    analytic_decision = state_logic(pointer, data)
    # Select state from Q-Table
    state_actions = q_table.iloc[select_state(pointer), :]
    # If the greedy factor is less than a randomly distributed number, if there are no values
    # on the Q-table, or if less than half the possible trades have been run without our trading logic,
    # return our analytical trade logic decision
    if np.random.uniform() > float(agent.epsilon) or state_actions.all() == 0 or pointer < int(TRADES_TO_RUN):
        return analytic_decision
    # Otherwise, return what has been working
    else:
        maximum = state_actions.idxmax()
        if str(maximum) == 'buy':
            return 0
        if str(maximum) == 'sell':
            return 1

# Selects the state on the Q-Table
def select_state(pointer):
    # Find the current price of the equity
    current_price = round(data['EQUITY'][pointer], 1)
    # Find the previous price of the equity
    previous_price = round(data['EQUITY'][pointer - 1], 1)
    if current_price > previous_price:
        return 0  # Equity Appreciated
    if current_price == previous_price:
        return 1  # Equity Held Value
    if current_price < previous_price:
        return 2  # Equity Depreciated

# Function to find the profit from trades
def determine_payoff(pointer, trade, inPortfolio):
    # Hold the value that the equity was purchased at
    global priceAtPurchase
    if inPortfolio:  # Stock is already owned
        if trade == 0:  # Cannot rebuy the equity; return delta
            #print('Holding Equity at $' + str(round(data['EQUITY'][pointer], 2)))
            #print('Purchase Price: $' + str(round(priceAtPurchase, 2)))
            inPortfolio = True
            return (0, inPortfolio)
        if trade == 1:  # Sell the Equity
            inPortfolio = False  # Remove Equity from portfolio
            #print('** Equity sold at $' + str(round(data['EQUITY'][pointer], 2)))
            return (data['EQUITY'][pointer] - priceAtPurchase, inPortfolio)
    if inPortfolio == False:  # Equity is not owned
        if trade == 0:  # Buy the equity
            inPortfolio = True  # Add it to the portfolio
            #print('** Equity bought at $' + str(round(data['EQUITY'][pointer], 2)))  # Display Price Equity was purchased at
            priceAtPurchase = data['EQUITY'][pointer]  # Record the price at which the Equity was purchased
            return (0.0, inPortfolio)
        if trade == 1:  # Sell
            inPortfolio = False
            #print('Out of the market at $' + str(round(data['EQUITY'][pointer], 2)))
            return (0.0, inPortfolio) 
 
# Runs RL script
def run(use_decay=False):
    # Builds the Q-Table
    q_table = build_q_table(STATES, ACTIONS)
    inPortfolio = False
    aggregate_profit = []
    # Assuming 0 profit -- or a portfolio with a reference of $0
    profit = 0
    # Move through all possible trades
    for x in range(2,TOTAL_TRADES):
        print(x, end='\r')
        # RL Agent chooses the trade
        trade = choose_trade(x - 1, q_table)
        # Find the payoff from the trade
        try:
            result, inPortfolio = determine_payoff(x, trade, inPortfolio)
        except TypeError:
            continue
        except:
            raise
        # Display to user
        #print('Profit from instance: ' + str(round(result, 2)))
        # Append result from trade to aggregate profit
        aggregate_profit.append(result)
        # Append to profit
        profit += result
        q_predict = q_table.iloc[select_state(x), trade]
        # If statement for last trade, tweak this
        if x == TOTAL_TRADES-1:
            q_target = result + float(agent.gamma) * q_table.iloc[select_state(x), :
                    ].max()
        else:
            q_target = result + float(agent.gamma) * q_table.iloc[select_state(x), :
                    ].max()
        # Append to located cell in Q-Table || Tweak this
        q_table.iloc[select_state(x), trade] += float(agent.alpha) * (q_target
                - q_predict)
        if use_decay and agent.epsilon >= EPSILON_DECAY:
            agent.epsilon = agent.epsilon - 0.005
        #print('\n')
    print()
    if inPortfolio:
        print("**** Please note that Equity is still held and may be traded later, this may affect profits ****")
    # Return the Q-Table and profit as a tuple
    profit = np.sum(aggregate_profit)
    return q_table, profit

# Ensures everything is loaded
if __name__ == '__main__':
    q_table, profit = run()
    print('''\r
        Q-table:
    ''')
    # Add reference column
    q_table["Reference"] = ['When Equity Appreciated', 'When Equity Held Value', 'When Equity Depreciated']
    print(q_table)
    # Show profits
    calc_profits = 1 + (round(profit, 2)/1000.0)
    calc_profits = calc_profits * float(STARTING_PORTFOLIO_VALUE)
    print('\nProfits from trading ' + str(GIVEN_EQUITY) + ' with starting portfolio of $' + str(STARTING_PORTFOLIO_VALUE) + ': $' + str(calc_profits))
