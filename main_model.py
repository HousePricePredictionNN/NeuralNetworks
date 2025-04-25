import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def house_price_prediction_using_mlp_model(filename, number_of_rows_to_load, number_of_epochs, input_test_data, loss_values_array, val_loss_array):
    # Rozpoczecie pomiaru czasu
    timer_start = time.time()

    # Wczytanie danych z pliku CSV
    data = pd.read_csv(filename, sep=',', nrows=number_of_rows_to_load)

    # Wypelnienie pustych pol metoda medianowa
    data.fillna(data.median(), inplace=True)

    X = data.drop(columns=['price'])
    y = data['price']

    # Podział na zbioru danych na zbior treningowy i testowy (80% - trening, 20% - test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Podział zbioru treningowego na treningowy i walidacyjny (80% - trening, 20% - walidacja)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Normalizacja danych (przeskalowanie cech, aby mialy srednia rowna 0 i wariancje 1)
    scaler_X = RobustScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    # Normalizacja etykiet (przy regresji czesto poprawia stabilnosc treningu)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).reshape(-1)
    y_val = scaler_y.transform(y_val.to_numpy().reshape(-1, 1)).reshape(-1)
    y_test = scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).reshape(-1)

    # Konwersja danych do tensorow
    X_train_tensor = torch.from_numpy(X_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    y_val_tensor = torch.from_numpy(y_val).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # Ustawienie urzadzenia
    device = torch.device("cpu")
    X_train_tensor = X_train_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    # Definicja sieci neuronowej (MLP - Multi Layer Perceptron)
    model = nn.Sequential(
        nn.Linear(X_train_tensor.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)

    # Definicja funkcji straty i optymalizatora
    loss_fn = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Petla treningowa z ewaluacja na zbiorze walidacyjnym
    for t in range(number_of_epochs):
        # Ustawienie modelu w tryb treningowy
        model.train()

        # Zerowanie gradientow
        optimizer.zero_grad()

        # Obliczanie predykcji i usuniecie zbednych wymiarow za pomoca .squeeze()
        y_pred_train = model(X_train_tensor).squeeze()

        # Obliczenie wartosci funkcji straty
        loss = loss_fn(y_pred_train, y_train_tensor)

        # Obliczenie gradientow
        loss.backward()

        # Aktualizacja wag
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val_tensor).squeeze()
            val_loss = loss_fn(y_pred_val, y_val_tensor)

        # Okresowa ewaluacja na zbiorze walidacyjnym (na poczatku, koncu i co setna epoke)
        if t % 100 == 0 or t == number_of_epochs - 1:
            # Ewaluacja modelu na zbiorze walidacyjnym
            if t == 0:
                print(f"first training epoch => train loss: {loss.item()}, val loss: {val_loss.item()}")
            elif t == number_of_epochs - 1:
                print(f"final training epoch => train loss: {loss.item()}, val loss: {val_loss.item()}")
            else:
                print(f"epoch: {t} => train loss: {loss.item()}, val loss: {val_loss.item()}")

        # Dodanie wartosci funkcji straty do tablicy
        val_loss_array.append(val_loss.item())

        # Dodanie wartosci funkcji straty do tablicy
        loss_values_array.append(loss.item())

    # Ewaluacja modelu na zbiorze testowym
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze()
        test_loss = loss_fn(test_predictions, y_test_tensor)
        print(f"\nfinal test loss: {test_loss.item()}\n")

    # Denormalizacja predykcji i prawdziwych wartosci dla zbioru testowego
    y_test_inv = scaler_y.inverse_transform(y_test_tensor.cpu().numpy().reshape(-1, 1))
    test_predictions_inv = scaler_y.inverse_transform(test_predictions.cpu().numpy().reshape(-1, 1))

    # Zakonczenie pomiaru czasu
    timer_stop = time.time()
    print(f"\nTime to finish: {timer_stop - timer_start:.2f} sec")

    # Obliczenie metryk i wyswietlenie metryk do oceny modelu
    mae = mean_absolute_error(y_test_inv, test_predictions_inv)
    rmse = mean_squared_error(y_test_inv, test_predictions_inv)
    r2 = r2_score(y_test_inv, test_predictions_inv)
    print(f"Model Scores:\nMAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")

    # Dane do oceny przez model, ktore pochodza spoza zbioru danych, ale sa prawdziwe
    verification_data = input_test_data

    # Skalowanie danych przykładowych
    verification_df = pd.DataFrame(verification_data, columns=X.columns)
    verification_df_scaled = scaler_X.transform(verification_df)

    # Konwersja zbioru weryfikujacego do tensor i predykcja modelu na zbiorze
    verification_tensor = torch.from_numpy(verification_df_scaled).float().to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(verification_tensor).cpu().numpy().reshape(-1, 1)

    # Denormalizacja predykcji
    predicted_data = scaler_y.inverse_transform(prediction)

    return predicted_data, loss_values_array, val_loss_array