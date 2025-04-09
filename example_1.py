# Importowanie bibliotek
import torch
import math

# Ustawienie typu danych na liczby zmiennoprzecinkowe i urzadzenia przetwarzajacego na CPU
dtype = torch.float
device = torch.device("cpu")


def simple_sinus_function_regression_using_polynomial(number_of_epochs):
    print(f"\nTotal number of epochs: {number_of_epochs}")

    # Utworzenie 2k rownomiernie ulozonych punktow od [-Pi; Pi] oraz funkcji sin(x), do ktorej bedziemy aproksymowac nasz wielomian
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Inicjalizacja losowych wag dla wspolczynnikow wielomianu: y = a + bx + cx^2 + dx^3
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(number_of_epochs):
        # Wyznaczenie wartosci naszego wielomianu z aktualnymi wagami
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Wyznaczenie MSE dla otrzymanego wielomianu w porownaniu do funkcji sinus
        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 0 and t != 0:
            print(f"epoch: {t} => loss: {loss}")
        elif t == number_of_epochs - 1:
            print(f"final epoch => loss: {loss}")
        elif t == 0:
            print(f"first epoch => loss: {loss}")

        # Reczne wyznaczenie gradientow poprzez obiczenie funkcji kosztu dla kazdej wagi (a-d)
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Atkutalizacja wag z wykorzystaniem wczesniej wyznaczonych gradientow
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    print(f'result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


# Wywolanie dla 2k epok
simple_sinus_function_regression_using_polynomial(2000)

# Wywolanie dla 5k epok
simple_sinus_function_regression_using_polynomial(5000)
