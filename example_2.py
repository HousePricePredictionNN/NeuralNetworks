# Importowanie bibliotek
import torch
import math

# Ustawienie typu danych na liczby zmiennoprzecinkowe i urzadzenia przetwarzajacego na CPU
dtype = torch.float
device = torch.device("cpu")


def simple_sinus_function_regression_using_polynomial_with_autograd(number_of_epochs):
    print(f"\nTotal number of epochs: {number_of_epochs}")

    # Utworzenie 2k rownomiernie ulozonych punktow od [-Pi; Pi] oraz funkcji sin(x), do ktorej bedziemy aproksymowac nasz wielomian
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Inicjalizacja losowych wag dla wspolczynnikow wielomianu: y = a + bx + cx^2 + dx^3
    # Dodatkowym atrybutem kazdej wagi jest od teraz requires_grad=True, ktore pozwala na sledzenie operacji na nich przez PyTorch
    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(number_of_epochs):
        # Wyznaczenie wartosci naszego wielomianu z aktualnymi wagami
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Wyznaczenie MSE dla otrzymanego wielomianu w porownaniu do funkcji sinus
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 0 and t != 0:
            print(f"epoch: {t} => loss: {loss.item()}")
        elif t == number_of_epochs - 1:
            print(f"final epoch => loss: {loss.item()}")
        elif t == 0:
            print(f"first epoch => loss: {loss.item()}")

        # Automatyczne wyznaczenie gradientow z wykorzystaniem backward
        loss.backward()

        # Wylaczenie sledzenia operacji na wagach za pomoca funckji no_grad()
        # Kazda waga zawiera wyznaczona wartosc gradientu (∂loss/∂waga) w polu grad
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            # Reczne zerowanie gradientow po aktualizacji wag
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

    print(f'result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


# Wywolanie dla 2k epok
simple_sinus_function_regression_using_polynomial_with_autograd(2000)

# Wywolanie dla 5k epok
simple_sinus_function_regression_using_polynomial_with_autograd(5000)
