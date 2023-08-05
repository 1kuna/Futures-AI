### Data Gathering and Storage
- Find an API that can pull live **FUTURES** market data (likely gonna be Yahoo).
- Determine the best time scales to pull data from.
    - Dunno if I should mix small with large or not (figure out the logistics).
- Determine options prices and correlate that with P/L. Might not be as huge a deal with futures as it would with stock options, not sure.
    - Need to find a paper trading API that has /ES futures available
- Find the best method for saving market data in realtime for training and fine-tuning. SQL, CSV, or something else?
- Use TA module for technicals but see if there's a module for support/resistance and Fibs. If one doesn't exist, create one.

### Data Cleaning
- Gather the data and store it in whatever the best format is determined to be.
- Deal with missing data in whatever way may be necessary, whether that be averaging, truncation, mean/median/mode, a model to predict missing values, or something else.
- Determine whether I need to standardize or normalize the data and do so if necessary.
    - If either one is necessary be sure it's able to handle outliers. Also make sure the rest of the model is made to handle outliers.
- Check for multicollinearity and deal with it if necessary.

### Feature Engineering
- After organizing the data, consider creating lagged features.
    - May not be necessary if I use RNNs or LSTMs, but it's worth considering.
- Do research on the importantce of interaction features and whether or not they're necessary in this case.

### Model Building
- After gathering the data, start with a rudimentary model test with something like matplotlib and scikit-learn.
    - Deduce the best general model for the data and make refinements to the data gathering and storage process if necessary.
    - After that, start building the actual model with Tensorflow, PyTorch, or Keras.
- Determine the best architecture for the model. Afterwards, do hyperparameter tuning. (Do more research into these)
- Be sure to check the model for over/underfitting. If it's overfitting, use regularization. If it's underfitting, add more data or increase the model's complexity. (Check to see if this ^^^ is true, source is GitHub Copilot)
- Do research into Vanishing and Exploding Gradients for RNNs and LSTMs. If it's an issue, use gradient clipping. (Research gradient clipping, also sourced from Copilot)
    - This point can improve performance

### Model Tuning
- Continue fine-tuning the model using online/incremental learning so it can continue to learn in real-time
- Will need to determine a timeframe for fine-tuning to be done. Every N data points.
    - Tools like Tensorflow's `partial_fit()`, Tensorflow Extended (TFX), or PyTorch can be used for this.
    - TFX can make a pipeline for the model to be trained on new data as it comes in.
    - Look into other tools for this as well.

### Model Evaluation
- Determine the best metric to test the model's performance on. (MSE, MAE, RMSE, etc.)
- Use out-of-sample data to test the model's performance to get a better idea of how it will perform in the real world.
- Track the model's performance over time to see if it's improving or not. If not, I might need to retrain the model and adjust the parameters.

### Trading Strategy
- Set clear rules for when the model should buy/sell/hold.
    - Could potentially utilize a reinforcement learning algorithm to determine the best trading strategy.
    - Could also base this on confidence probabilities.
- Determine the best way to manage risk. (Stop-loss, take-profit, etc.)
- Decide a profit-taking strategy, whether that be a trailing stop-loss, a fixed take-profit, when the model predicts a reversal, or something else.
- Determine the best way to manage position size. (Fixed, Kelly Criterion, etc.)
- Be wary of the fact that futures have been troublesome for entering limit positions (in the paper account anyway) so look into market orders and if they're possible.
- Quantify performance in a method other than P/L. (Sharpe Ratio, Sortino Ratio, etc.)

### Additional Notes
- Could implement sentiment analysis with a pre-trained language model.

- Could use a web framework like Flask or Django to create a web app that can manage the model and display the performance of all the processes.
- Could also use a database like MongoDB to store the data, will need to research this more.