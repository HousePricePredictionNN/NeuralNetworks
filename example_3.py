# Importowanie bibliotek
import torch
import math

# Ustawienie typu danych na liczby zmiennoprzecinkowe i urzadzenia przetwarzajacego na CPU
dtype = torch.float
device = torch.device("cpu")


def simple_sinus_function_regression_using_polynomial_with_nn(number_of_epochs):
    print(f"\nTotal number of epochs: {number_of_epochs}")

    # Utworzenie 2k rownomiernie ulozonych punktow od [-Pi; Pi] oraz funkcji sin(x), do ktorej bedziemy aproksymowac nasz wielomian
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Inicjalizacja tensora, który reprezentuje współczynniki wielomianu trzeciego stopnia
    p = torch.tensor([1, 2, 3])

    # Utworzenie macierzy reprezentujacej wielomian trzeciego stopnia postaci: xx[i] = [x_i, x_i^2, x_i^3]
    xx = x.unsqueeze(-1).pow(p)

    # Utworzenie modelu, poprzez sekwencje kolenych operacji:
    #   - .nn.Linear(3, 1) => wykorzystuje funkcje liniowa do wyznaczenia wag i bias, odpowiednik y_pred = bias + w1x + w2x^2 + w3x^3
    #   - .nn.Flatten(0, 1) => "splaszcza" wyjscie otrzymane z poprzedniej operacji do wyznaczonego wymiaru, w tym przypadku do wymiaru odpowiadajacemu 'y'
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
    )

    # Definiowanie funkcji straty na MSE, uzywany w poprzednich przykladach i definiowany recznie
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-6
    for t in range(number_of_epochs):
        # Wyznaczenie wartosci naszego wielomianu z wykorzystaniem modelu
        y_pred = model(xx)

        # Wyznaczenie wartosci funkcji straty z wykorzstaniem wczesniejszej definicji
        loss = loss_fn(y_pred, y)
        if t % 100 == 0 and t != 0:
            print(f"epoch: {t} => loss: {loss}")
        elif t == number_of_epochs - 1:
            print(f"final epoch => loss: {loss}")
        elif t == 0:
            print(f"first epoch => loss: {loss}")

        # Zerowanie gradientow, poniewaz inaczej skumulowalyby sie
        model.zero_grad()

        # Automatyczne wyznaczenie gradientow z wykorzystaniem backward
        loss.backward()

        # Aktualizacja wag kazdego z parametrow znajdujacego sie w modelu (nasz 3-elementowy tensor)
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    # Bezposredni dostep do 1. warstwy modelu
    linear_layer = model[0]

    print(
        f'result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')


# Wywolanie dla 2k epok
simple_sinus_function_regression_using_polynomial_with_nn(2000)

# Wywolanie dla 5k epok
simple_sinus_function_regression_using_polynomial_with_nn(5000)
