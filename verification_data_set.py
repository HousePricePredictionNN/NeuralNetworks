import pandas as pd

verification_data = pd.read_csv('verification_data.csv', sep=',')

verification_data_set = verification_data.drop(columns=['price'])
verification_data_set_expected_prices = verification_data['price']