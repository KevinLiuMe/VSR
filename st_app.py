import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.interpolate import SmoothBivariateSpline
import os

st. set_page_config(
    layout="wide")

# Initialize session state for tracking if the simulation has been run
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False

# Streamlit form for inputs

if not st.session_state.run_simulation:
    st.title('Volatility Analysis for Stocks')
    with st.form(key='stock_form'):
        stock_symbol = st.text_input('Stock Symbol', 'AAPL')
        start_date = st.date_input('Start Date', value=datetime(2022, 1, 1))
        end_date = st.date_input('End Date', value=datetime.now())
        submit_button = st.form_submit_button(label='Run Simulation')
        
        if submit_button:
            st.session_state.run_simulation = True
            st.session_state.stock_symbol = stock_symbol
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
            st.rerun()  # Refresh the app to show results

if st.session_state.run_simulation:
    stock_symbol = st.session_state.stock_symbol
    st.title(f'Volatility Surface of {stock_symbol}')
    st.write(f'A dashboard to visualize volatility surface associated with {stock_symbol} constructed with options data from yfinance')
    st.markdown("---")
    st.write("")

    col1, col2, col3 = st.columns([15, 1, 15])
    with col3:
        today = datetime.now()

        # Get the Ticker object for the stock
        stock = yf.Ticker(stock_symbol)


        stock_price_data = stock.history(period='5d')

        # Check if the DataFrame is empty
        if stock_price_data.empty:
            st.error(f"No data available for {stock_symbol}.")

        stock_price = stock.history(period='5d')['Close'].iloc[-1]

        # Fetch the available expiration dates from yfinance
        expiration_dates = stock.options

        all_options = []

        # Initialize lists to store strike, IV, and days to expiry separately
        all_strikes = []
        all_IVs = []
        all_days_to_expiry = []
        all_log_moneyness = []
        all_types = []  # Store the type of each option (Call or Put)

        # Loop through each expiration date
        for exp_date in expiration_dates:
            try:
                # Get option chain data for the expiration date
                opt = stock.option_chain(exp_date)

                # Extract calls and puts
                calls = opt.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].assign(type='Call', expiry=exp_date).query('impliedVolatility > 0.01')
                puts = opt.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].assign(type='Put', expiry=exp_date).query('impliedVolatility > 0.01')

                # Combine calls and puts
                options = pd.concat([calls, puts])

                # Extract relevant data
                strikes = options['strike'].tolist()  # Strike prices
                IVs = options['impliedVolatility'].tolist()  # Implied volatilities
                types = options['type'].tolist()  # Option type (Call/Put)

                # Calculate days to expiration
                exp_date_datetime = pd.to_datetime(exp_date)  # Convert expiration date string to datetime
                days_to_expiry = [(exp_date_datetime - today).days] * len(strikes)

                log_moneyness = [np.log(stock_price / strike) if option_type == 'Call' else np.log(strike / stock_price) for strike, option_type in zip(strikes, types)]

                # Append to lists
                all_strikes.extend(strikes)
                all_IVs.extend(IVs)
                all_days_to_expiry.extend(days_to_expiry)
                all_log_moneyness.extend(log_moneyness)
                all_options.append(options)
                all_types.extend(types)

            except Exception as e:
                st.write(f"Error retrieving options for {exp_date}: {e}")
        
        if len(all_days_to_expiry) == len(all_IVs) == len(all_log_moneyness):
            # Ensure there are no NaN or infinite values
            df = pd.DataFrame({
                'days_to_expiry': all_days_to_expiry,
                'implied_volatility': [iv * 100 for iv in all_IVs],  # Convert IV to percentage
                'log_moneyness': all_log_moneyness,
                'type': all_types  # Call or Put type
            }).dropna()

        # Check for empty or too-small datasets
            if len(df) < 3:
                st.error("Not enough data points to create a surface. Try fetching more data or adjust the stock symbol.")
            else:
                # Create a grid for interpolation
                yi = np.linspace(df['days_to_expiry'].min(), df['days_to_expiry'].max(), 100)
                xi = np.linspace(df['log_moneyness'].min()-0.1, df['log_moneyness'].max(), 100)
                yi, xi = np.meshgrid(yi, xi)

                # Linear interpolation to smooth the data
                zi_linear = griddata(
                    (df['days_to_expiry'], df['log_moneyness']),
                    df['implied_volatility'],
                    (yi, xi),
                    method='linear'
                )

                # Use 'nearest' interpolation to fill NaN values
                zi_nearest = griddata(
                    (df['days_to_expiry'], df['log_moneyness']),
                    df['implied_volatility'],
                    (yi, xi),
                    method='nearest'
                )

                # Fill NaNs in the linear interpolation with nearest-neighbor results
                zi = np.where(np.isnan(zi_linear), zi_nearest, zi_linear)

                # Apply Gaussian smoothing to make the surface even smoother (optional)
                zi_smoothed = gaussian_filter(zi, sigma=1)  # Apply Gaussian smoothing with a sigma of 1

                # Create the surface plot using Plotly's Surface
                fig = go.Figure()

                # Add the smoothed surface
                fig.add_trace(go.Surface(
                    x=xi,  # Days to expiry grid (X-axis)
                    y=yi,  # Log moneyness grid (Y-axis)
                    z=zi_smoothed,  # Smoothed interpolated implied volatility surface (Z-axis)
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(
                        len=0.7,  # Shorten the color bar to 50% of its usual height
                        thickness=15,  # Adjust thickness to make it a bit more compact
                        yanchor="bottom",  # Anchor the color bar to the top
                        y = 0.3,
                        title='IV (%)'  # Add a title to the color bar
                    ),
                ))

                # Add scatter points for call options (green)
                call_options = df[df['type'] == 'Call']
                fig.add_trace(go.Scatter3d(
                    y=call_options['days_to_expiry'],
                    x=call_options['log_moneyness'],
                    z=call_options['implied_volatility'],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color='green',
                        symbol='circle',
                        opacity=0.5
                    ),
                    name="CALL"
                ))

                # Add scatter points for put options (orange)
                put_options = df[df['type'] == 'Put']
                fig.add_trace(go.Scatter3d(
                    y=put_options['days_to_expiry'],
                    x=put_options['log_moneyness'],
                    z=put_options['implied_volatility'],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color='orange',
                        symbol='circle',
                        opacity=0.5,
                    ),
                    name="PUT"
                ))


                # Update layout for the plot
                fig.update_layout(
                    title=f'Smoothed Volatility Surface with Data Points for {stock_symbol.upper()}',
                    scene=dict(
                        xaxis_title='Log Moneyness',
                        yaxis_title='Days to Expiry',
                        zaxis_title='Implied Volatility (%)',
                    ),
                    width=1000,
                    height=600,
                    showlegend=True,  # Ensure legend is visible
                    legend=dict(
                        y=0.3,  # Move the legend lower, adjust this as necessary
                        yanchor='top',  # Align the bottom of the legend to the position
                    )
                )

        else:
            st.error("Data length mismatch: Check the input data dimensions.")
        if all_options:
            options_df = pd.concat(all_options).reset_index(drop=True)
            options_df.index.name = None

            curr_time = datetime.utcnow()
            hour = curr_time.hour
            minute = curr_time.minute
            day = curr_time.day
            month = curr_time.month

            # Displaying Options Data
            st.write(f'{stock_symbol} Options data updated at {hour:02}:{minute:02}, 2024-{month:02}-{day:02} UTC:')
            st.dataframe(options_df.set_index(options_df.columns[0]), height=300, width=1200)



        
            

            





    with col1:
        # Placeholder for the left half of the screen
        st.write("Implied volatility (IV) represents the market's forecast of how much the price of an option's underlying asset is \
                    expected to fluctuate over the option's lifespan. IV varies among options for the same underlying asset. \
                    Generally, higher IV leads to higher option premiums, while lower IV results in lower premiums.")
        st.write("")
        st.write("Studying volatility surfaces helps traders and investors understand how the market's expectations of volatility \
                    vary across different strikes and maturities for options. By analyzing these surfaces, one can identify mispricings \
                    and inefficiencies in the options market. This knowledge allows for the development of trading strategies that exploit \
                    discrepancies between implied volatility and actual market volatility. Essentially, it provides insights into how volatility \
                    expectations change over time and across different strike prices, which can be crucial for effective risk management, pricing \
                    options, and identifying potential arbitrage opportunities.")    
    
    st.markdown("---")

    col1, col2, col3 = st.columns([15, 1, 15])
    with col1:
        st.markdown('# Binomial Trees')
        st.write(f"For this model, we will utilize binomial trees to approximate implied volatility for {stock_symbol}. At each time step, the stock price can either move up or down. The stock price at each node is determined by:")
        st.latex(r"S_{i,j} = S_0 \times u^j \times d^{i-j}")
        st.markdown("""
        **Where:**

        - $S_{i,j}$: Stock price at time step $i$ after $j$ upward movements.
        - $u = e^{\sigma \sqrt{\Delta t}}$: Upward price movement factor.
        - $d = \\frac{1}{u} = e^{-\sigma \sqrt{\Delta t}}$: Downward price movement factor.
        """)



        st.write("The risk-neutral probability of an upward movement is given by:")
        st.latex(r"p = \frac{e^{r \Delta t} - d}{u - d}")
        st.write("we choose $$ {e^{r \Delta t} - d} $$ rather than $$ {1+r-d} $$ to better capture countinuous compounding of risk free interest rate over time $$ {\Delta t} $$ as opposed to simple interest.")

    with col3: 
        st.markdown('#')
        st.write("To calculate the option's value at earlier nodes, we work backward through the tree. The option value at each node is the discounted expected value of the option in the next time step:")
        st.latex(r"V_{i,j} = e^{-r \Delta t} \left( p \cdot V_{i+1, j+1} + (1 - p) \cdot V_{i+1, j} \right)")

        st.write("By working backwards from the final time step to the first, we calculate the option's value at \( t = 0 \), which is the current option price:")
        st.latex(r"V_0 = e^{-r \Delta t} \left( p \cdot V_u + (1 - p) \cdot V_d \right)")
        st.markdown("Where $V_u$ and $V_d$ are the values of the option if the stock moves up or down, respectively.")
        st.write("In this model, we will utilize the [Newton-Raphson Method of root finding](https://en.wikipedia.org/wiki/Newton%27s_method) to interatively guess IV until it matches with the current option prices.")
        st.write("We will set the threshold $$ {\epsilon} $$ for Vega at 1e-4 to ensure convergence by switching to the [bisection method](https://en.wikipedia.org/wiki/Bisection_method).")

    st.markdown("---")
    
    col1, col2, col3 = st.columns([15, 1, 15])
    with col1:
        st.markdown('# Linear Interpolation')
        st.write("for our first attempt we construct the volatility surface using a [linear interpolation method](https://en.wikipedia.org/wiki/Linear_interpolation) with [nearest-neighbor interpolation](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation) to cover NaN values.")
        st.write("")
        st.write("For an option with strike price \( K \) and underlying spot price \( S \), we define log moneyness \( k \) as:")

        st.latex(r'''
        k =
        \begin{cases} 
        \log\left(\frac{S}{K}\right) & \text{if Call} \\ 
        \log\left(\frac{K}{S}\right) & \text{if Put}
        \end{cases}
        ''') 
        st.write("Sample data:")
        st.dataframe(df,height=275, width=850)
        
    

    with col3:
         # Debugging: Check if there are enough points for interpolation
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Number of data points: {len(df)}")
    
    st.markdown("---")
