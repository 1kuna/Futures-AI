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