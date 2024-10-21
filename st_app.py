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
from scipy.interpolate import Rbf
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline

st.set_page_config(page_title="Volatility Surface Visualizer", page_icon="💡", layout="wide")

if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False


if not st.session_state.run_simulation:
    # Center the title
    st.markdown("<h1 style='text-align: center;'>Volatility Analysis for Stocks</h1>", unsafe_allow_html=True)

    # Create empty columns to center the form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form(key='stock_form'):
            stock_symbol = st.text_input('Stock Symbol', 'MSFT')
            submit_button = st.form_submit_button(label='Run Simulation')
            
            if submit_button:
                st.session_state.run_simulation = True
                st.session_state.stock_symbol = stock_symbol
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

        if stock_price_data.empty:
            st.error(f"No data available for {stock_symbol}.")

        stock_price = stock.history(period='5d')['Close'].iloc[-1]

        expiration_dates = stock.options
        all_options = []
        all_strikes = []
        all_IVs = []
        all_days_to_expiry = []
        all_log_moneyness = []
        all_types = []

        for exp_date in expiration_dates:
            try:
                # Get option chain data for the expiration date
                opt = stock.option_chain(exp_date)

                calls = opt.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].assign(type='Call', expiry=exp_date).query('impliedVolatility > 0.01')
                puts = opt.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].assign(type='Put', expiry=exp_date).query('impliedVolatility > 0.01')
                calls = calls.query('volume >= 10')
                puts = puts.query('volume >= 10')
                options = pd.concat([calls, puts])

                # Extract relevant data
                strikes = options['strike'].tolist()
                IVs = options['impliedVolatility'].tolist()
                types = options['type'].tolist()

                # Calculate days to expiration
                exp_date_datetime = pd.to_datetime(exp_date)  # Convert expiration date string to datetime
                days_to_expiry = [(exp_date_datetime - today).days] * len(strikes)

                log_moneyness = [np.log(stock_price / strike) if option_type == 'Call' else np.log(strike / stock_price) for strike, option_type in zip(strikes, types)]

                all_strikes.extend(strikes)
                all_IVs.extend(IVs)
                all_days_to_expiry.extend(days_to_expiry)
                all_log_moneyness.extend(log_moneyness)
                all_options.append(options)
                all_types.extend(types)

            except Exception as e:
                st.write(f"Error retrieving options for {exp_date}: {e}")

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
        if len(all_days_to_expiry) == len(all_IVs) == len(all_log_moneyness):
            df = pd.DataFrame({
                'days_to_expiry': all_days_to_expiry,
                'implied_volatility': [iv * 100 for iv in all_IVs],  # Convert IV to percentage
                'log_moneyness': all_log_moneyness,
                'type': all_types  # Call or Put type
            }).dropna()
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

        st.write("We filter by Implied Volatility greater than 0.01 to avoid options with little market activity and Volume greater than or equal to 10 to exclude illiquid options.")
        st.write("We also focus on the middle ninety percentile of all data-points to prevent the effects of extreme outliers on our data set")
        st.write("Sample data:")
        st.dataframe(df,height=250, width=850)
        
    

    with col3:
        if len(all_days_to_expiry) == len(all_IVs) == len(all_log_moneyness):
            df = pd.DataFrame({
                'days_to_expiry': all_days_to_expiry,
                'implied_volatility': [iv * 100 for iv in all_IVs],  # Convert IV to percentage
                'log_moneyness': all_log_moneyness,
                'type': all_types  # Call or Put type
            }).dropna()


            max_days = 180

                # Filter for days_to_expiry <= max_days
            df = df[df['days_to_expiry'] <= max_days]

            log_moneyness_min = df['log_moneyness'].quantile(0.05)
            log_moneyness_max = df['log_moneyness'].quantile(0.95)
            days_to_expiry_min = df['days_to_expiry'].quantile(0.05)
            days_to_expiry_max = df['days_to_expiry'].quantile(0.95)

            df = df[
                (df['log_moneyness'] >= log_moneyness_min) &
                (df['log_moneyness'] <= log_moneyness_max) &
                (df['days_to_expiry'] >= days_to_expiry_min) &
                (df['days_to_expiry'] <= days_to_expiry_max)
            ]



            if len(df) < 3:
                st.error("Not enough data points to create a surface. Try fetching more data or adjust the stock symbol.")
            else:
                yi = np.linspace(df['days_to_expiry'].min(), df['days_to_expiry'].max(), 100)
                xi = np.linspace(df['log_moneyness'].min(), df['log_moneyness'].max(), 100)
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
                zi_smoothed = gaussian_filter(zi, sigma=1)

                fig = go.Figure()

                    # Add the smoothed surface
                fig.add_trace(go.Surface(
                    x=xi,  # Days to expiry grid (X-axis)
                    y=yi,  # Log moneyness grid (Y-axis)
                    z=zi_smoothed,  
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(
                        len=0.8,  
                        thickness=15,  
                        yanchor="bottom", 
                        y = 0.3,
                        title='IV (%)'
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
                    width=800,
                    height=800,
                    showlegend=True,  # Ensure legend is visible
                    legend=dict(
                        y=0.3,  # Move the legend lower, adjust this as necessary
                        yanchor='top',  # Align the bottom of the legend to the position
                    )
                )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")


    col1, col2, col3 = st.columns([15, 1, 15])
    with col3:
        if len(all_days_to_expiry) == len(all_IVs) == len(all_log_moneyness):
            # Ensure there are no NaN or infinite values
            df = pd.DataFrame({
                'days_to_expiry': all_days_to_expiry,
                'implied_volatility': [iv * 100 for iv in all_IVs],  # Convert IV to percentage
                'log_moneyness': all_log_moneyness,
                'type': all_types  # Call or Put type
            }).dropna()

            # Filter implied volatilities <= 500 and days to expiry <= 180
            df_filtered = df[(df['implied_volatility'] <= 500) & (df['days_to_expiry'] <= 180)]

            # Create a fine grid for log-moneyness and days to expiry
            log_moneyness_grid = np.linspace(df_filtered['log_moneyness'].quantile(0.05), df_filtered['log_moneyness'].quantile(0.95), 50)
            days_to_expiry_grid = np.linspace(df_filtered['days_to_expiry'].quantile(0.05), df_filtered['days_to_expiry'].quantile(0.95), 50)
            log_moneyness_grid, days_to_expiry_grid = np.meshgrid(log_moneyness_grid, days_to_expiry_grid)

            # Perform linear interpolation for 'days_to_expiry'
            IV_linear = griddata(
                (df_filtered['log_moneyness'], df_filtered['days_to_expiry']),
                df_filtered['implied_volatility'],
                (log_moneyness_grid, days_to_expiry_grid),
                method='linear'
            )

            # Slider for epsilon value in Gaussian smoothing
            epsilon_value = 0.01

            # Define the Gaussian RBF for log moneyness
            rbf_gaussian = Rbf(df_filtered['log_moneyness'], df_filtered['days_to_expiry'], df_filtered['implied_volatility'],
                            function='gaussian', epsilon=epsilon_value)

            # Use the RBF function to predict implied volatility on the grid
            IV_gaussian = rbf_gaussian(log_moneyness_grid, days_to_expiry_grid)

            # Replace NaN values from linear interpolation with Gaussian smoothed values
            IV_combined = np.where(np.isnan(IV_linear), IV_gaussian, IV_linear)

            # Cap the smoothed values at a maximum of 500 to avoid unrealistic spikes
            IV_combined = np.clip(IV_combined, 0, 500)

            # Create the 3D surface plot with Plotly
            fig_rbf = go.Figure()

            fig_rbf.add_trace(go.Surface(
                x=log_moneyness_grid,  # Log-Moneyness on X-axis
                y=days_to_expiry_grid,  # Days to Expiry on Y-axis
                z=IV_combined,  # Combined implied volatility on Z-axis
                colorscale='Viridis',
                opacity=0.7,
                showscale=True,
                colorbar=dict(
                    len=0.8,
                    thickness=15,
                    yanchor="bottom",
                    y=0.3,
                    title='IV (%)'
                ),
            ))

            log_moneyness_min = df_filtered['log_moneyness'].quantile(0.05)
            log_moneyness_max = df_filtered['log_moneyness'].quantile(0.95)
            days_to_expiry_min = df_filtered['days_to_expiry'].quantile(0.05)
            days_to_expiry_max = df_filtered['days_to_expiry'].quantile(0.95)

            df_filtered = df_filtered[
                (df_filtered['log_moneyness'] >= log_moneyness_min) &
                (df_filtered['log_moneyness'] <= log_moneyness_max) &
                (df_filtered['days_to_expiry'] >= days_to_expiry_min) &
                (df_filtered['days_to_expiry'] <= days_to_expiry_max)
            ]

            call_options = df_filtered[df_filtered['type'] == 'Call']
            fig_rbf.add_trace(go.Scatter3d(
                x=call_options['log_moneyness'],
                y=call_options['days_to_expiry'],
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

            # **Add scatter points for put options (orange)**
            put_options = df_filtered[df_filtered['type'] == 'Put']
            fig_rbf.add_trace(go.Scatter3d(
                x=put_options['log_moneyness'],
                y=put_options['days_to_expiry'],
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


            # Layout settings for the plot
            fig_rbf.update_layout(
                title='Hybrid Volatility Surface: Linear for Days to Expiry, Gaussian for Moneyness',
                scene=dict(
                    xaxis_title='Log Moneyness',
                    yaxis_title='Days to Expiry',
                    zaxis_title='Implied Volatility (%)',
                ),
                width=800,
                height=800,
                showlegend=True,  # Ensure legend is visible
                legend=dict(
                        y=0.3,  # Move the legend lower, adjust this as necessary
                        yanchor='top',  # Align the bottom of the legend to the position
                    )
            )

        st.plotly_chart(fig_rbf, use_container_width=True)

        
    with col1:
        st.markdown('# Kernel Smoothing')

        st.write("""
        Gaussian smoothing applies a kernel inspired by the normal distribution to assign weights to data points based on their distance
        from the point being estimated. The Gaussian kernel ensures that closer points (in terms of strike price and time to maturity)
        are weighted more heavily in the volatility estimate.
        """)
        st.write("Our choice of kernel function ϕ is Gaussian-like and inspired by the industry standard 'TradFi' data vendor [OptionMetrics](https://optionmetrics.com/).")
        # Gaussian kernel function using st.latex
        st.latex(r"""
        \phi(x, y) = \frac{1}{2\pi} \exp\left( -\frac{x^2}{2h_x} - \frac{y^2}{2h_y} \right)
        """)

        st.write("""
        Where:
        - \( x \) and \( y \) are the differences between the grid point and the option in terms of log moneyness and time to maturity.
        - \( h_x \) and \( h_y \) are the smoothing parameters (bandwidths) that control how much influence nearby points have.
        """)
        st.write("""
        Epsilon controls the smoothness of the kernel function in both log moneyness and time to maturity.
        It acts as the bandwidth parameter for the kernel function, determining the range of influence of data points.
        """)

        st.write("At each grid point ( j ) on the volatility surface, the smoothed volatility is computed as:")

        st.latex(r"""
        \hat{\sigma}_j = \frac{\sum_{i=1}^{n} \sigma_i^{mark} w_i \phi(x_{ij}, y_{ij})}{\sum_{i=1}^{n} w_i \phi(x_{ij}, y_{ij})}
        """)

    st.markdown("---")

    with st.spinner("Constructing SVI surface and plotting..."):
        df = pd.DataFrame({
                    'days_to_expiry': all_days_to_expiry,
                    'implied_volatility': [iv * 100 for iv in all_IVs],  # Convert IV to percentage
                    'log_moneyness': all_log_moneyness,
                    'type': all_types  # Call or Put type
                }).dropna()

        log_moneyness_min = df['log_moneyness'].quantile(0.05)
        log_moneyness_max = df['log_moneyness'].quantile(0.95)
        days_to_expiry_min = df['days_to_expiry'].quantile(0.05)
        days_to_expiry_max = df['days_to_expiry'].quantile(0.95)

        df= df[
                    (df['log_moneyness'] >= log_moneyness_min) &
                    (df['log_moneyness'] <= log_moneyness_max) &
                    (df['days_to_expiry'] >= days_to_expiry_min) &
                    (df['days_to_expiry'] <= days_to_expiry_max)
                ]


        df['years_to_expiry'] = df['days_to_expiry'] / 365
        df['total_variance'] = (df['implied_volatility'] ** 2) * df['years_to_expiry']

        # Step 2: Define the SVI Functions
        def raw_svi(k, a, b, rho, m, sigma):
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        def fit_svi_parameters(log_moneyness, total_variance):
            def svi_objective(params):
                a, b, rho, m, sigma = params
                model_variance = raw_svi(log_moneyness, a, b, rho, m, sigma)
                return model_variance - total_variance

            initial_guess = [0.1, 0.1, 0.0, 0.0, 0.1]
            bounds = (
                [-np.inf, 0, -1, -np.inf, 0],
                [np.inf, np.inf, 1, np.inf, np.inf]
            )

            result = least_squares(
                svi_objective,
                x0=initial_guess,
                bounds=bounds,
                max_nfev=10000
            )

            return result.x

        grouped = df.groupby('days_to_expiry')

        svi_params = {}

        for expiry, group in grouped:
            log_moneyness = group['log_moneyness'].values
            total_variance = group['total_variance'].values

            # Fit SVI parameters for this expiry
            try:
                params = fit_svi_parameters(log_moneyness, total_variance)
                svi_params[expiry] = params
            except Exception as e:
                st.write(f"Could not fit SVI parameters for expiry {expiry}: {e}")


        log_moneyness_grid = np.linspace(df['log_moneyness'].min(), df['log_moneyness'].max(), 100)
        days_to_expiry_grid = sorted(svi_params.keys())

        implied_vol_surface = np.zeros((len(days_to_expiry_grid), len(log_moneyness_grid)))

        for i, expiry in enumerate(days_to_expiry_grid):
            a, b, rho, m, sigma = svi_params[expiry]
            total_variance = raw_svi(log_moneyness_grid, a, b, rho, m, sigma)
            years_to_expiry = expiry / 365
            implied_vol_surface[i, :] = np.sqrt(total_variance / years_to_expiry)

        fig = go.Figure(data=[go.Surface(
            x=log_moneyness_grid,
            y=days_to_expiry_grid,
            z=implied_vol_surface,
            colorscale='Viridis',
            colorbar=dict(
                        len=0.8,  
                        thickness=15,  
                        yanchor="bottom", 
                        y = 0.3,
                        title='IV (%)'
                    ),
        )])

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

        fig.update_layout(
            title='SVI Volatility Surface',
            scene=dict(
                xaxis_title='Log Moneyness',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility'
            ),
            width=800,
            height=800,
            showlegend=True,  # Ensure legend is visible
            legend=dict(
                        y=0.3,  # Move the legend lower, adjust this as necessary
                        yanchor='top',  # Align the bottom of the legend to the position
                    )
            
        )

        col1, col2, col3 = st.columns([15, 1, 15])
        with col3 : 
            st.plotly_chart(fig, use_container_width=True)

        with col1 :
            st.markdown('# Stochastic Volatility Inspired (SVI) Parameterization')

            st.write("""
            The Stochastic Volatility Inspired (SVI) parameterization of the volatility surface is designed for the calibration of a time-specific slice of the implied volatility surface, capturing the dynamics of the skew/smile across strikes. 
            This can be challenging to model using traditional approaches. 
            """)

            # Add a mathematical formula using st.latex
            st.latex(r"""
            f(k; \chi) = a + b \left( \rho(k + m) + \sqrt{(k - m)^2 + s^2} \right)
            """)

            st.write("The implied volatility for a grid point \( j \) is given by:")

            st.latex(r"""
            \hat{\sigma}_j = \frac{f(k_j; \chi)}{\sqrt{\tau_j}}
            """)

            st.write("""
            For each time slice , we solve a non-linear weighted least squares problem to fit the SVI parameters:

            """)

            st.latex(r"""
            \min_{\chi_{\tau}} \sum_{i=1}^{n} w_i \left( \hat{\sigma}_i - \sigma_i^{\text{mark}} \right)^2
            """)

            st.write("""

            Note that our calibration of the SVI surface does not guarantee the absence of static (calendar and butterfly) arbitrage. 
            Interpolation over various time slices is performed with quadratic bivariate spline approximations.
            """)

    st.markdown("---")

    st.markdown(
        "[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white&color=rgba(0%2C0%2C0%2C0))](https://github.com/KevinLiuMe/Volatility-Dashboard)"
        )
