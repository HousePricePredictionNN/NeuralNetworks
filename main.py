from main_plots import *
from main_model import *
from verification_data_set import *

# Parametry poczatkowe
loss_array = []
csv_filename = 'data.csv'
number_of_rows_to_load_from_csv_file = 15000
number_of_epochs = 5000

# Uruchomienie modelu sieci neuronwej
house_pricing_prediction, loss_array = house_price_prediction_using_mlp_model(csv_filename,
                                                                              number_of_rows_to_load_from_csv_file,
                                                                              number_of_epochs, verification_data_set,
                                                                              loss_array)
house_pricing_prediction = house_pricing_prediction.flatten()

# Prezentacja wynikow
print("\nPrice predictions for example input data:\n-----------------------------------------")
for i, (predicted_price, real_price) in enumerate(zip(house_pricing_prediction, verification_data_set_expected_prices)):
    percentage_diff = ((predicted_price - real_price) / real_price) * 100
    sign = "+" if percentage_diff >= 0 else "-"

    pred_str = f"{predicted_price:,.0f}".replace(",", " ")
    real_str = f"{real_price:,.0f}".replace(",", " ")

    print(
        f"Prediction: {pred_str} zł\nReal: \t\t{real_str} zł\nPercentage difference: \t\t{sign}{abs(percentage_diff):.2f}%\n-----------------------------------------")

plot_loss_curve(loss_array)
plot_predictions_vs_actual(verification_data_set_expected_prices, house_pricing_prediction)
