# Reflection-Coefficient-PINN

## Постановка задачи

Рассматривается тонкая упругая пластина, колебания которой описываются дифференциальным уравнением:
$$(\Delta^2 - \rho h \omega^2/D)\zeta(x,y) = F(x,y),$$
где
- $\Delta$ — двумерный оператор Лапласа,
- $\rho$ — плотность материала,
- $h$ — толщина пластины,
- $\omega$ — частота колебаний,
- $D$ — изгибная жесткость пластины,
- $F$ — внешняя гармоническая сила.

Коэффициенты отражения и прохождения для системы из двух периодических массивов резонаторов выражаются как
$$R = |R_0|^2 = \left|\frac{i\mu}{4a\kappa^3 \sin{\psi_0}}\left(\zeta_1 e^{i k_y \delta y_1} + \zeta_2 e^{i k_y \delta y_2}\right)\right|^2, \quad T = |T_0|^2 = |1 + R_0|^2,$$
$$\zeta_1 = \frac{(1-\mu S)e^{i\kappa_y \delta y_1} + \mu S_1e^{i\kappa_y \delta y_2}}{(1 - \mu S)^2 - \mu^2 S_1S_2}, \quad \zeta_2 = \frac{(1 - \mu S)e^{i\kappa_y \delta y_2} + \mu S_2 e^{i\kappa_y \delta y_1}}{(1 - \mu S)^2 - \mu^2 S_1S_2},$$
$$S = \frac{1}{4a\kappa^3}\sum\limits_{n=-\infty}^{+\infty} \left(\lambda_n^{-1} - \gamma_n^{-1}\right),$$
$$S_1 = \frac{1}{4a\kappa^3}\sum\limits_{n=-\infty}^{+\infty} e^{ik_x (\delta x_1 - \delta x_2)}\left(\frac{e^{-\kappa\lambda_n|\delta y_1 - \delta y_2|}}{\lambda_n} - \frac{e^{-\kappa \gamma_n |\delta y_1 - \delta y_2|}}{\gamma_n}\right),$$
$$S_2 = \frac{1}{4a\kappa^3}\sum\limits_{n=-\infty}^{+\infty} e^{ik_x (\delta x_2 - \delta x_1)}\left(\frac{e^{-\kappa\lambda_n|\delta y_2 - \delta y_1|}}{\lambda_n} - \frac{e^{-\kappa \gamma_n |\delta y_2 - \delta y_1|}}{\gamma_n}\right),$$
$$\lambda_n = \sqrt{(k_{xn}/\kappa)^2 - 1}, \quad \gamma_n = \sqrt{(k_{xn}/\kappa)^2 + 1}, \quad k_{xn} = k_x + 2\pi n /a.$$

## Входные данные

| Обозначение | Параметр |
| - | - |
| $\omega$ | Частота |
| $\psi$ | Угол падения |
| $\delta x$ | Продольный сдвиг |
| $\delta y$ | Поперечный сдвиг |
| $\mathrm{Re}(\mu)$ | Действительная часть поляризуемости |
| $\mathrm{Im}(\mu)$ | Мнимая часть поляризуемости |

## Выходные данные

| Обозначение | Параметр |
| - | - |
| $\mathrm{Re}(\zeta_1)$ | Действительная часть амплитуды первой цепочки |
| $\mathrm{Im}(\zeta_1)$ | Мнимая часть амплитуды первой цепочки |
| $\mathrm{Re}(\zeta_2)$ | Действительная часть амплитуды второй цепочки |
| $\mathrm{Im}(\zeta_2)$ | Мнимая часть амплитуды второй цепочки |