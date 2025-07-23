from src.ElijahA.preprocessing import load_data, preprocess_data, save_preprocessed_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    # Define file paths
    input_file = 'data/your_data.csv'  # Adjust path as necessary
    output_file = 'data/preprocessed_data.csv'

    # Load data
    print("Loading data...")
    data = load_data(input_file)

    # Preprocess the data
    print("Preprocessing data...")
    processed_data = preprocess_data(data)

    # Save the preprocessed data
    print("Saving the preprocessed data...")
    save_preprocessed_data(processed_data, output_file)

    # Split processed data into features and target variable
    X = processed_data.drop(columns=['sold_price'])  # Adjust based on your target column
    y = processed_data['sold_price']

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a machine learning model
    print("Training the model...")
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

if __name__ == '__main__':
    main()
