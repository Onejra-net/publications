---

title: "AI - Wstęp"
date: 2026-10-13
category: Article
description: "AI - Wstęp"
---

# Wstęp do matematyki używanej przy opisie AI

Ten tekst to **wprowadzenie**, nie pełny kurs. Ma dać intuicję i zapis formalny pojęć, które pojawiają się przy opisie małych modeli językowych (embedding, attention, MLP, strata, gradienty).

Zakładamy poziom: działania na liczbach, potęgi, funkcja wykładnicza i logarytm naturalny. Resztę budujemy od zera.

---

## 1. Skalar, wektor, macierz

**Skalar** — pojedyncza liczba, np. $3{,}14$ albo strata $\mathcal{L}$.

**Wektor** — uporządkowana lista liczb o ustalonej długości $d$:

$$
\mathbf{x} = (x_0, x_1, \ldots, x_{d-1}) \in \mathbb{R}^d.
$$

Przykład przy $d=3$: $\mathbf{x}=(-0{,}03,\;0{,}08,\;-0{,}23)$.

**Macierz** — tablica liczb o $m$ wierszach i $n$ kolumnach:

$$
A \in \mathbb{R}^{m \times n}, \qquad
A = \begin{pmatrix}
A_{00} & A_{01} & \cdots & A_{0,n-1} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m-1,0} & \cdots & \cdots & A_{m-1,n-1}
\end{pmatrix}.
$$

W AI często:

| obiekt | sens |
|--------|------|
| wiersz macierzy embeddingów | wektor jednego tokena |
| macierz wag $W$ | uczy się przekształcać wektory |

---

## 2. Dodawanie wektorów

Dodajemy **składowa po składowej** (wektory muszą mieć tę samą długość):

$$
(\mathbf{a}+\mathbf{b})_k = a_k + b_k.
$$

Przykład (embedding + pozycja):

$$
\mathbf{x}_t = \mathbf{e}_{u_t} + \mathbf{p}_t.
$$

---

## 3. Iloczyn skalarny (dot product)

Dla $\mathbf{a},\mathbf{b}\in\mathbb{R}^d$:

$$
\mathbf{a}\cdot\mathbf{b}
= \sum_{k=0}^{d-1} a_k\,b_k.
$$

Intuicja: miara „podobnego kierunku”. Duży dodatni wynik → wektory wskazują w podobną stronę.

W attention score to właśnie iloczyn skalarny $\mathbf{q}\cdot\mathbf{k}$ (potem dzielony przez $\sqrt{d}$).

---

## 4. Mnożenie wektor × macierz

Niech $\mathbf{x}\in\mathbb{R}^{d}$ oraz $W\in\mathbb{R}^{d\times n}$. Wynik $\mathbf{y}\in\mathbb{R}^{n}$:

$$
y_j = \sum_{i=0}^{d-1} x_i\,W_{ij}, \qquad j=0,\ldots,n-1.
$$

To **projekcja liniowa**: ten sam $\mathbf{x}$ przez różne $W$ daje Q, K, V albo wejście do MLP.

Zapis skrótowy w tekstach o AI:

$$
\mathbf{y} = \mathbf{x}^\top W
\quad\text{(w zależności od konwencji wierszy/kolumn)}.
$$

W programie edukacyjnym liczymy wprost sumę jak wyżej.

---

## 5. Funkcje nieliniowe: $\tanh$, $\exp$, $\log$

Sieć bez nieliniowości to tylko składanie mnożeń macierzy (nadal „jedna” wielka liniowość). Dlatego pojawiają się funkcje:

**Tangens hiperboliczny** (w MLP):

$$
\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}} \in (-1,1).
$$

**Eksponenta** $e^{z}$ — w softmaxie.

**Logarytm naturalny** $\log$ — w stracie $-\log\pi_{y}$ (gdy $\pi_y$ rośnie, strata maleje).

---

## 6. Softmax

Z wektora „surowych” ocen $\boldsymbol{\ell}=(\ell_0,\ldots,\ell_{V-1})$ robimy **rozkład prawdopodobieństwa**:

$$
\pi_v
= \frac{e^{\ell_v}}{\sum_{v'=0}^{V-1} e^{\ell_{v'}}},
\qquad
\pi_v \ge 0, \quad \sum_v \pi_v = 1.
$$

Stabilny zapis numeryczny (ten sam wynik matematyczny):

$$
\pi_v
= \frac{e^{\ell_v - m}}{\sum_{v'} e^{\ell_{v'}-m}},
\quad m=\max_{v'}\ell_{v'}.
$$

Użycie: wagi uwagi w attention oraz prawdopodobieństwa następnego tokena na wyjściu.

---

## 7. Strata (loss) przy klasyfikacji tokena

Model przewiduje rozkład $\boldsymbol{\pi}$ po całym vocabularzu. Prawdziwy następny znak ma indeks $y$.

**Cross-entropy** (u nas dla jednego przykładu):

$$
\mathcal{L} = -\log \pi_y.
$$

- Jeśli $\pi_y \to 1$, to $\mathcal{L}\to 0$ (dobrze).
- Jeśli $\pi_y$ jest małe, $\mathcal{L}$ jest duże (źle).

To skalar — od niego zaczyna się liczenie gradientów.

---

## 8. Pochodna i reguła łańcuchowa (idea autogradu)

**Pochodna** mówi, jak bardzo wyjście zmienia się przy małej zmianie wejścia.

Dla złożenia $z=f(g(x))$:

$$
\frac{dz}{dx} = \frac{dz}{dg}\cdot\frac{dg}{dx}.
$$

W sieci $\mathcal{L}$ zależy od tysięcy operacji. **Autograd** idzie od $\mathcal{L}$ wstecz i dla każdej operacji stosuje lokalną regułę, np.:

| operacja | reguła (szkic) |
|----------|----------------|
| $c=a+b$ | $\partial\mathcal{L}/\partial a \mathrel{+}= \partial\mathcal{L}/\partial c$ (to samo dla $b$) |
| $c=a\cdot b$ | $\partial\mathcal{L}/\partial a \mathrel{+}= b\,\partial\mathcal{L}/\partial c$ |
| $c=\tanh(a)$ | $\partial\mathcal{L}/\partial a \mathrel{+}= (1-\tanh^2 a)\,\partial\mathcal{L}/\partial c$ |

Na końcu każdy parametr $\theta$ ma liczbę $\partial\mathcal{L}/\partial\theta$ — **gradient**.

---

## 9. Spadek gradientowy (SGD)

Uczenie w najprostszej postaci:

$$
\theta \leftarrow \theta - \eta\,\frac{\partial\mathcal{L}}{\partial\theta}.
$$

- $\eta>0$ — **learning rate** (u nas np. $0{,}025$),
- znak minus: idziemy w stronę **zmniejszania** $\mathcal{L}$.

Za duże $\eta$ — niestabilność; za małe — bardzo wolna nauka.

---

## 10. Co z tego składa się w „mini-GPT”

Kolejność pojęć w modelu znakowym:

1. **Token** → indeks w vocabularzu  
2. **Embedding** $E$ + **pozycja** $P$ → wektor $\mathbf{x}_t$  
3. **Q, K, V** — trzy projekcje liniowe $\mathbf{x}_t$  
4. **Attention** — iloczyny skalarne, softmax, ważona suma $\mathbf{v}$  
5. **MLP** — mnożenia przez $W_1,W_2$ + $\tanh$ + residual  
6. **Logity** → **softmax** → $\boldsymbol{\pi}$  
7. **$\mathcal{L}=-\log\pi_y$** → **backward** → **SGD**

Każdy z tych kroków używa wyłącznie narzędzi z punktów 1–9.

---

## 11. Minimalny słownik symboli

| symbol | znaczenie |
|--------|-----------|
| $\mathbb{R}^d$ | wektory długości $d$ |
| $d$ / $d_{\mathrm{model}}$ | szerokość reprezentacji |
| $d_h$ / $d_{\mathrm{hidden}}$ | szerokość ukrytej warstwy MLP |
| $V$ | rozmiar vocabularza |
| $T$ | długość kontekstu |
| $\eta$ | learning rate |
| $\mathbf{x},\mathbf{q},\mathbf{k},\mathbf{v},\mathbf{h}$ | wektory aktywacji |
| $E,P,W,\mathbf{b}$ | parametry (wagi) |
| $\boldsymbol{\pi}$ | rozkład prawdopodobieństwa |
| $\mathcal{L}$ | strata (skalar) |

---

*To jest warstwa „matematyka pod spodem”. Opis konkretnego przebiegu obliczeń na zdaniu (np. krótkim przykładzie znakowym) buduje się już z tych klocków.*
