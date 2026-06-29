from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def split_and_scale_data(df, scale=True):
    df = df.copy()
    X = df.drop(columns=['source_x', 'source_y', 'I_0'])
    y = df[['source_x', 'source_y', 'I_0']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_train_loc = y_train[['source_x', 'source_y']]
    y_test_loc = y_test[['source_x', 'source_y']]
    y_train_int = y_train['I_0']
    y_test_int = y_test['I_0']

    X_test_unscaled = X_test.copy()

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, y_train_loc, y_test_loc, y_train_int, y_test_int, X_test_unscaled