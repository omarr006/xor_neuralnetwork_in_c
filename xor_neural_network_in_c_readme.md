# XOR Neural Network in C

2-input neural network in C, trained on XOR.

## Was drin ist

* **Forward Pass:** Berechnung der Aktivierungen durch Hidden- und Output-Layer.

* **Backpropagation:** Gradientenabstieg zur Anpassung der Gewichte und Biases.

* **Sigmoid:** Aktivierungsfunktion $\sigma(x)$ inklusive erster Ableitung.

* **Trainingsschleife:** Epochenweises Training mit Mean Squared Error (MSE) als Verlustfunktion.

## Bauen & Starten

Kompilieren und direkt ausführen:

```
gcc main.c -o nn -lm && ./nn

```

## Beispielausgabe

```
Epoch     0 | MSE Loss: 0.134482
Epoch  5000 | MSE Loss: 0.000496
Epoch 10000 | MSE Loss: 0.000171
Epoch 15000 | MSE Loss: 0.000102
Epoch 20000 | MSE Loss: 0.000073
Epoch 25000 | MSE Loss: 0.000056

---final guess!---
In: [0, 0] -> Target: 0 | Predicted: 0.0096 (round: 0)
In: [0, 1] -> Target: 1 | Predicted: 0.9900 (round: 1)
In: [1, 0] -> Target: 1 | Predicted: 0.9899 (round: 1)
In: [1, 1] -> Target: 0 | Predicted: 0.0124 (round: 0)

```

## Nächste Schritte

* **Mehr Layer / Dynamische Topologie:** Flexible Matrix-Strukturen für beliebig viele Hidden-Layer einbinden.

  …