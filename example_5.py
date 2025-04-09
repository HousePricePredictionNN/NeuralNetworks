# Importowanie bibliotek
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def house_price_prediction_using_mlp_model(number_of_epochs, example):
    # Pomiar czasu wykonania
    start = time.time()

    # Wczytanie datasetu housing - domy w USA
    housing = fetch_california_housing()
    X = housing.data  # macierz cech (features)
    y = housing.target  # ceny mieszkań (target)

    # Podzial na zbior treningowy i testowy (80% - trening, 20% - test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizacja danych (przeskalowanie cech, aby mialy srednia rowną 0 i wariancje 1)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Normalizacja etykiet (przy regresji czesto poprawia stabilnosc treningu)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    # Konwersja danych do tensora
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # Ustawienie urzadzenia
    device = torch.device("cpu")
    X_train_tensor = X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    # Definicja sieci neuronowej (MLP - Multi Layer Perceptron)
    # Wejscie ma tyle neuronow ile cech w zestawie danych (housing - 13 cech)
    model = nn.Sequential(
        nn.Linear(X_train_tensor.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Softplus()
    ).to(device)

    # Definicja funkcji straty i optymalizatora
    loss_fn = nn.MSELoss()
    learning_rate = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Petla treningowa
    for t in range(number_of_epochs):
        # Ustawienie modelu w tryb treningowy
        model.train()

        # Zerowanie gradientow
        optimizer.zero_grad()

        # Obliczanie predykcji i usuniecie zbednych wymiarow za pomoca .squeeze()
        y_pred = model(X_train_tensor).squeeze()

        # Obliczenie wartosci funkcji straty
        loss = loss_fn(y_pred, y_train_tensor)
        if t % 100 == 0 and t != 0:
            print(f"epoch: {t} => loss: {loss.item()}")
        elif t == number_of_epochs - 1:
            print(f"final training epoch => loss: {loss.item()}")
        elif t == 0:
            print(f"first training epoch => loss: {loss.item()}")

        # Obliczenie gradientow
        loss.backward()

        # Aktualizacja wag
        optimizer.step()

    # Ewaluacja modelu na zbiorze testowym
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze()
        test_loss = loss_fn(test_predictions, y_test_tensor)
        print(f"\nfinal test loss: {test_loss.item()}\n")

    # Denormalizacja predykcji i prawdziwych wartosci
    y_test_inv = scaler_y.inverse_transform(y_test_tensor.cpu().numpy().reshape(-1, 1))
    test_predictions_inv = scaler_y.inverse_transform(test_predictions.cpu().numpy().reshape(-1, 1))

    # Obliczenie metryk
    mae = mean_absolute_error(y_test_inv, test_predictions_inv)
    rmse = mean_squared_error(y_test_inv, test_predictions_inv)
    r2 = r2_score(y_test_inv, test_predictions_inv)

    print("Model Scores:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

    end = time.time()
    print(f"\nTime to finish: {end - start:.2f} sec")

    # Przykladowe dane do oceny przez model
    example_input = example

    # Skalowanie danych
    example_input_scaled = scaler_X.transform(example_input)

    # Konwersja i predykcja
    example_tensor = torch.from_numpy(example_input_scaled).float().to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(example_tensor).cpu().numpy().reshape(-1, 1)

    # Denormalizacja
    prediction_real = scaler_y.inverse_transform(prediction)

    print(f"\nPrice prediction for example input data: {100 * prediction_real[0][0]:.3f} (in $1,000)")


# Przyklad przykladowych danych (wszystkie cechy recznie ustawione)
# Nazwy cech: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# mediana dochodow, wiek budynku, liczba pokoi, liczba sypialni, liczba mieszkancow, srednia liczba mieszkancow w pojedynczym domostwie, szerokosc geograficzna, dlugosc geograficzna
example = np.array([[5.0, 30.0, 5.0, 1.0, 1000.0, 3.0, 34.0, -118.0]])

house_price_prediction_using_mlp_model(5000, example)
